"""Run kernel regression stage.

Copied from Peng's MSCA-284-copy/fit-kreg.py script
Modified based on Alex's MSCA-286-tune-kronreg/2024_06_12/run_kronreg.ipynb

"""

from typing import Callable

import fire
import pandas as pd
from kreg.kernel.component import KernelComponent
from kreg.kernel.factory import (
    build_exp_similarity_kfunc,
    build_gaussianRBF_kfunc,
    build_matern_three_half_kfunc,
    vectorize_kfunc,
)
from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import BinomialLikelihood
from kreg.model import KernelRegModel
from onemod.schema.stages import KregModel
from onemod.utils import Subsets, get_handle


def build_kernel(model_config: KregModel) -> KroneckerKernel:
    """_summary_

    # TODO: flexibility for different kernels

    Parameters
    ----------
    model_config : KregModel
        _description_

    Returns
    -------
    list[Callable]
        _description_

    """
    location_kernel = build_exp_similarity_kfunc(model_config.exp_location)
    age_kernel = build_gaussianRBF_kfunc(model_config.gamma_age)
    year_kernel = build_matern_three_half_kfunc(model_config.gamma_year)
    kernel_components = [
        KernelComponent(
            ["super_region_id", "region_id", "location_id"],
            vectorize_kfunc(location_kernel),
        ),
        KernelComponent(["transformed_age"], vectorize_kfunc(age_kernel)),
        KernelComponent(["year_id"], vectorize_kfunc(year_kernel)),
    ]
    return KroneckerKernel(kernel_components, nugget=model_config.nugget)


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

    # Zero test data, rescale age, add offset
    # TODO: generalize offset for different model types
    index = data.eval(f"{config.test} == 1")
    data.loc[index, config.obs] = 0.0
    data.loc[index, config.weights] = 0.0
    age_scale = model_config.age_scale
    data["transformed_age_mid"] = data.eval(
        "log(exp(@age_scale * age_mid) - 1) / @age_scale"
    )
    data.loc[data.eval(f"{config.pred} == 0"), config.pred] = 1e-10
    data["offset"] = data.eval(f"log({config.pred} / (1 - {config.pred}))")

    # Create and fit kernel regression model
    kernel = build_kernel(model_config)
    likelihood = BinomialLikelihood(config.obs, config.weights, "offset")
    model = KernelRegModel(kernel, likelihood, lam=model_config.lam)
    data["kreg_y"], history = model.fit(**stage_config.kreg_fit.model_dump())
    print(history)

    # Create predictions
    data["kreg"] = data.eval("1 / (1 + exp(-(kreg_y + offset)))")

    # Save results
    dataif.dump_kreg(model, f"submodels/{submodel_id}/model.pkl")
    dataif.dump_kreg(
        data[config.ids + ["kreg", "kreg_y", "offset"]].rename(
            columns={"kreg": config.pred}
        ),
        f"submodels/{submodel_id}/predictions.parquet",
    )


def main() -> None:
    fire.Fire(kreg_model)
