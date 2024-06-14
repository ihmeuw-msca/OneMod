"""Run kernel regression stage.

Copied from Peng's MSCA-284-copy/fit-kreg.py script

"""

from typing import Callable

import fire
import jax.numpy as jnp
import pandas as pd
from kreg.kernel.factory import (
    build_exp_similarity_kfunc,
    build_gaussianRBF_kfunc,
    build_RQ_kfunc,
    vectorize_kfunc,
)
from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import BinomialLikelihood
from kreg.model import KernelRegModel
from numpy.typing import NDArray
from onemod.schema import OneModConfig
from onemod.schema.stages import KregModel
from onemod.utils import Subsets, get_handle


def build_kernels(
    data: pd.DataFrame, model_config: KregModel
) -> list[Callable]:
    """_summary_

    # TODO: different kernels

    Parameters
    ----------
    data : pandas.DataFrame
        _description_
    model_config : KregModel
        _description_

    Returns
    -------
    list[Callable]
        _description_

    """
    kage_rbf = build_gaussianRBF_kfunc(model_config.gamma_age)

    # kage_linear = shifted_scaled_linear_kernel(
    #     data["transformed_age_mid"].mean(), data["transformed_age_mid"].std()
    # )

    @vectorize_kfunc
    def age_kernel(x, y):
        return kage_rbf(x, y)  # + 0.2 * kage_linear(x, y) + 0.2

    kyear_rq = build_RQ_kfunc(model_config.alpha_year, model_config.gamma_year)
    # kyear_rq = get_gaussianRBF(stage_config.gamma_year)
    # kyear_linear = shifted_scaled_linear_kernel(
    #     data["year_id"].mean(), data["year_id"].std()
    # )

    @vectorize_kfunc
    def year_kernel(x, y):
        return kyear_rq(x, y)  # + 0.2 * kyear_linear(x, y) + 0.2

    location_kernel = vectorize_kfunc(
        build_exp_similarity_kfunc(model_config.exp_location)
    )

    return [location_kernel, age_kernel, year_kernel]


def build_grids(data: pd.DataFrame) -> list[NDArray]:
    """_summary_

    # TODO: generalize for different dimensions

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
    config: OneModConfig, data: pd.DataFrame
) -> BinomialLikelihood:
    """_summary_

    Parameters
    ----------
    config : OneModConfig
        _description_
    data : pandas.DataFrame
        _description_

    Returns
    -------
    BinomialLikelihood
        _description_

    """
    obs_rate = jnp.asarray(data[config.obs].to_numpy())
    sample_size = jnp.asarray(data[config.weights].to_numpy())
    offset = jnp.asarray(data["offset"].to_numpy())

    index = data.eval(f"{config.test} == 0").to_numpy()
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
    model_config = stage_config.kreg_model

    # Load data and filter by subset
    # TODO: Generalize columns used
    subsets = Subsets(
        "kreg", stage_config, subsets=dataif.load_kreg("subsets.csv")
    )
    data = subsets.filter_subset(
        data=pd.merge(
            left=dataif.load_spxmod("predictions.parquet"),
            right=dataif.load_data(
                columns=config.ids
                + [
                    "super_region_id",
                    "region_id",
                    "age_mid",
                    config.obs,
                    config.weights,
                    config.test,
                ]
            ),
            on=config.ids,
            how="left",
        ),
        subset_id=int(submodel_id.removeprefix("subset")),
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
    a = model_config.age_scale
    data["transformed_age_mid"] = data.eval("log(exp(@a * age_mid) - 1) / @a")
    data["offset"] = data.eval(f"log({config.pred} / (1 - {config.pred}))")
    # TODO: generalize offset for different model types

    # Build kernels, etc., etc.
    kernels = build_kernels(data, model_config)
    grids = build_grids(data)
    kernel = KroneckerKernel(kernels, grids, nugget=model_config.nugget)
    likelihood = build_likelihood(config, data)

    # Create and fit kernel regression model
    model = KernelRegModel(kernel, likelihood, model_config.lam)
    data["kreg_y"], history = model.fit(**stage_config.kreg_fit.model_dump())
    print(history)

    # Create predictions
    data["kreg"] = data.eval("1 / (1 + exp(-(kreg_y + offset)))")

    # Save results
    # TODO: keep kreg_y, offset, anything else?
    dataif.dump_kreg(model, f"submodels/{submodel_id}/model.pkl")
    dataif.dump_kreg(
        data[config.ids + ["kreg", "kreg_y", "offset"]].rename(
            columns={"kreg": config.pred}
        ),
        f"submodels/{submodel_id}/predictions.parquet",
    )


def main() -> None:
    fire.Fire(kreg_model)
