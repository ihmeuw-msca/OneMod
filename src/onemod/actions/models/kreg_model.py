"""Run kernel regression stage.

Copied from Peng's MSCA-284-copy/fit-kreg.py script

"""

from typing import Callable

import fire
import jax.numpy as jnp
import pandas as pd
from kreg.kernel.factory import (
    get_exp_similarity_kernel,
    get_gaussianRBF,
    get_RQ_kernel,
    vectorize_kfunc,
)
from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import BinomialLikelihood
from kreg.model import KernelRegModel
from numpy.typing import NDArray
from onemod.schema.onemod import OneModConfig
from onemod.utils import Subsets, get_handle


def build_kernels(data: pd.DataFrame, config: dict) -> list[Callable]:
    """_summary_

    # TODO: Ask about commented code, generalize for different kernels, finish docstring

    Parameters
    ----------
    data : pandas.DataFrame
        _description_
    config : dict
        _description_

    Returns
    -------
    list[Callable]
        _description_

    """
    kage_rbf = get_gaussianRBF(config["gamma_age"])

    # kage_linear = shifted_scaled_linear_kernel(
    #     data["transformed_age_mid"].mean(), data["transformed_age_mid"].std()
    # )

    @vectorize_kfunc
    def age_kernel(x, y):
        return kage_rbf(x, y)  # + 0.2 * kage_linear(x, y) + 0.2

    kyear_rq = get_RQ_kernel(config["alpha_year"], config["gamma_year"])
    # kyear_rq = get_gaussianRBF(config["gamma_year"])
    # kyear_linear = shifted_scaled_linear_kernel(
    #     data["year_id"].mean(), data["year_id"].std()
    # )

    @vectorize_kfunc
    def year_kernel(x, y):
        return kyear_rq(x, y)  # + 0.2 * kyear_linear(x, y) + 0.2

    location_kernel = vectorize_kfunc(
        get_exp_similarity_kernel(config["exp_location"])
    )

    return [location_kernel, age_kernel, year_kernel]


def build_grids(data: pd.DataFrame) -> list[NDArray]:
    """_summary_

    # TODO: Generalize, finish docstring

    Parameters
    ----------
    data : pandas.DataFrame
        _description_

    Returns
    -------
    list[NDArray]
        _description_

    """
    labels = [
        ["super_region_id", "region_id", "location_id"],
        ["transformed_age_mid"],
        ["year_id"],
    ]

    data = data.sort_values(
        labels[0] + labels[1] + labels[2], ignore_index=True
    )  # FIXME: This is already done elsewhere

    grids = [data[label].drop_duplicates().to_numpy() for label in labels]
    for i, grid in enumerate(grids):
        if grid.shape[1] == 1:
            grids[i] = grid.ravel()

    return grids


def build_likelihood(
    config: OneModConfig, data: pd.DataFrame, train: str
) -> BinomialLikelihood:
    """_summary_

    # TODO: finish docstring

    Parameters
    ----------
    config : OneModConfig
        _description_
    data : pandas.DataFrame
        _description_
    train : str
        _description_

    Returns
    -------
    BinomialLikelihood
        _description_

    """
    obs_rate = jnp.asarray(data[config.obs].to_numpy())
    sample_size = jnp.asarray(data[config.weights].to_numpy())
    offset = jnp.asarray(data["offset"].to_numpy())

    index = data.eval(train).to_numpy()  # FIXME: use test column
    obs_rate = jnp.where(index, obs_rate, 0.0)
    sample_size = jnp.where(index, sample_size, 0.0)

    likelihood = BinomialLikelihood(obs_rate, sample_size, offset)
    return likelihood


def kreg_model(directory: str, submodel_id: str) -> None:
    """Run the kernel regression stage.

    Parameters
    ----------
    directory
        Experiment directory.
        - ``directory / config / settings.yml`` contains model settings
        - ``directory / results / kreg`` stores kreg results
    submodel_id
        Submodel to run based on groupby settings. For example, the
        submodel_id ``subset0`` corresponds to the data subset ``0``
        stored in ``directory / results / kreg / subsets.csv``.

    Outputs
    -------
    model.pkl
        Kreg model instance for diagnostics.
    predictions.parquet
        Submodel predictions.

    """
    dataif, config = get_handle(directory)
    stage_config = config.kreg

    # Load data and filter by subset
    # TODO: Generalize columns used
    subsets = Subsets(
        "kreg", stage_config, subsets=dataif.load_kreg("subsets.csv")
    )
    subset_id = int(submodel_id.removeprefix("subset"))
    data = pd.merge(
        left=subsets.filter_subset(
            dataif.load_spxmod("predictions.parquet"), subset_id
        ),
        right=dataif.load_data(
            columns=config.ids
            + ["super_region_id", "region_id", "age_mid", config.weights]
        ),
        on=config.ids,
        how="left",
    ).sort_values(
        by=[
            "super_region_id",
            "region_id",
            "location_id",
            "age_mid",
            "year_id",
        ],
        ignore_index=True,
    )
    a = 0.1  # TODO: Put into schema?
    data["transformed_age_mid"] = data.eval("log(exp(@a * age_mid) - 1) / @a")
    data["offset"] = data.eval(
        "log(@config.pred / (1 - @config.pred))"
    )  # TODO: does this work with @ and schema object?
    data["train"] = data[config.test] == 0

    # Build kernels, etc., etc.
    kernels = build_kernels(data, config)
    grids = build_grids(data)
    kernel = KroneckerKernel(
        kernels, grids, nugget=float(config["nugget"])
    )  # FIXME: might not need "float" because schema already casts as float
    likelihood = build_likelihood(config, data, "train")

    # Create and fit kernel regression model
    # TODO: Put fit arguments into schema?
    model = KernelRegModel(kernel, likelihood, config["lam"])
    data["kreg_y"], history = model.fit(
        gtol=5e-4,
        max_iter=20,
        cg_maxiter=100,
        cg_maxiter_increment=0,
        nystroem_rank=10,
    )
    print(history)

    # Create predictions
    data["kreg"] = data.eval("1 / (1 + exp(-(kreg_y + offset)))")

    # Save results
    dataif.dump_kreg(model, f"submodels/{submodel_id}/model.pkl")
    dataif.dump_kreg(data, f"submodels/{submodel_id}/predictions.parquet")


def main() -> None:
    fire.Fire(kreg_model)
