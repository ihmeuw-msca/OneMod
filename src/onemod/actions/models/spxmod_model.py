"""Run spxmod stage.

This stage fits a model with the covariates selected in the rover stage,
using priors to smooth covariate coefficients across age groups (based
on the 'age_mid' column in the input data). Users can also specify
intercepts and spline variables that vary by dimensions such as age
and/or location.

"""

from functools import partial
from typing import Callable

import fire
import numpy as np
import pandas as pd
from loguru import logger
from pplkit.data.interface import DataInterface
from spxmod.model import XModel
from xspline import XSpline

from onemod.modeling.residual import ResidualCalculator
from onemod.schema import OneModConfig
from onemod.utils import Subsets, get_handle


def get_coef(model: XModel) -> pd.DataFrame:
    """Get coefficient information from the fitted model.

    Parameters
    ----------
    model
        The statistical model object containing coefficient data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing coefficient, dimension, and dimension
        value information.

    """
    df_coef = []
    for var_builder in model.var_builders:
        df_sub = var_builder.space.span.copy()
        df_sub["cov"] = var_builder.name
        df_coef.append(df_sub)
    df_coef = pd.concat(df_coef, axis=0, ignore_index=True)
    df_coef["coef"] = model.core.opt_coefs
    return df_coef


def _get_covs(
    dataif: DataInterface, config: OneModConfig, subset: pd.DataFrame
) -> list[str]:
    """Get spxmod model covariates."""
    # Get covariates selected in previous stage; filter by subset
    selected_covs = dataif.load_rover_covsel("selected_covs.csv")
    for col_name in config.groupby:
        col_val = subset[col_name].item()
        selected_covs = selected_covs.query(f"{col_name} == {repr(col_val)}")
    selected_covs = selected_covs["cov"].tolist()

    # Get fixed covariates
    fixed_covs = config.rover_covsel.rover.cov_fixed
    if "intercept" in fixed_covs:
        fixed_covs.remove("intercept")
    return selected_covs + fixed_covs


def _get_spline_basis(column: pd.Series, spline_config: dict) -> pd.DataFrame:
    """Get spline basis based on data and configuration."""
    col_min, col_max = column.min(), column.max()
    spline_config["knots"] = col_min + np.array(spline_config["knots"]) * (
        col_max - col_min
    )
    spline = XSpline(**spline_config)
    idx_start = 0 if spline_config["include_first_basis"] else 1
    spline_basis = pd.DataFrame(
        spline.design_mat(column),
        columns=[
            f"spline_{ii+idx_start}" for ii in range(spline.num_spline_bases)
        ],
    )
    return spline_basis


def _add_selected_covs(xmodel_args: dict, selected_covs: list[str]) -> dict:
    """Add selected covariates to spxmod model configuration."""
    # add age_mid to spaces if not already included
    space_keys = [space["name"] for space in xmodel_args["spaces"]]
    if "age_mid" not in space_keys:
        xmodel_args["spaces"].append(
            dict(name="age_mid", dims=[dict(name="age_mid", type="numerical")])
        )

    # add variables for selected covs if not already included
    var_builder_keys = [
        (var_builder["name"], var_builder["space"])
        for var_builder in xmodel_args["var_builders"]
    ]
    for cov in selected_covs:
        if (cov, "age_mid") not in var_builder_keys:
            xmodel_args["var_builders"].append(dict(name=cov, space="age_mid"))

    return xmodel_args


def _build_xmodel_args(config: OneModConfig, selected_covs: list[str]) -> dict:
    """Format config data for spxmod xmodel.

    Model automatically includes a coefficient for each of the selected
    covariates and age group (based on the 'age_mid' column in the input
    data). Users can also specify intercepts and spline variables that
    vary by dimensions such as age and/or location.

    """
    xmodel_args = config.spxmod.xmodel.model_dump()
    coef_bounds = xmodel_args.pop("coef_bounds")
    lam = xmodel_args.pop("lam")

    xmodel_args = _add_selected_covs(xmodel_args, selected_covs)

    # default settings for everyone
    for var_builder in xmodel_args["var_builders"]:
        cov = var_builder["name"]
        if "uprior" not in var_builder or var_builder["uprior"] is None:
            var_builder["uprior"] = coef_bounds.get(cov)

        if "lam" not in var_builder or var_builder["lam"] is None:
            var_builder["lam"] = lam

    xmodel_args["model_type"] = config.mtype
    xmodel_args["obs"] = config.obs
    xmodel_args["weights"] = config.weights

    return xmodel_args


def spxmod_model(directory: str, submodel_id: str) -> None:
    """Run spxmod stage.

    This stage fits a model with the covariates selected in the rover
    stage, using priors to smooth covariate coefficients across age
    groups (based on the 'age_mid' column in the input data). Users can
    also specify intercepts and spline variables that vary by dimensions
    such as age and/or location.

    Parameters
    ----------
    directory
        Parent folder where the experiment is run.
        - ``directory / config / settings.yml`` contains model settings
        - ``directory / results / spxmod`` stores spxmod results
    submodel_id
        Submodel to run based on groupby setting. For example, the
        submodel_id ``subset0`` corresponds to the data subset ``0``
        stored in ``directory / results / spxmod / subsets.csv``.

    Outputs
    -------
    model.pkl
        SPxMod model instance for diagnostics.
    coef.csv
        Model coefficients for different age groups.
    predictions.parquet
        Predictions with residual information.

    """
    dataif, config = get_handle(directory)
    stage_config = config.spxmod

    # Load data, filter by subset, and add spline basis
    subsets = Subsets(
        "spxmod", stage_config, subsets=dataif.load_spxmod("subsets.csv")
    )
    subset_id = int(submodel_id.removeprefix("subset"))
    df = subsets.filter_subset(dataif.load_data(), subset_id)
    if stage_config.xmodel.spline_config is not None:
        spline_config = stage_config.xmodel.spline_config.model_dump()
        col_name = spline_config.pop("name")
        df = pd.concat(
            [df, _get_spline_basis(df[col_name], spline_config)], axis=1
        )
    df_train = df.query(f"({config.test} == 0) & {config.obs}.notnull()")

    # Get spxmod model covariates
    selected_covs = _get_covs(
        dataif, config, subsets.subsets.query(f"subset_id == {subset_id}")
    )
    logger.info(f"Running spxmod with covariates: {selected_covs}")

    # Create spxmod model parameters
    xmodel_args = _build_xmodel_args(config, selected_covs)
    xmodel_fit_args = stage_config.xmodel_fit
    logger.info(
        f"{len(xmodel_args['var_builders'])} var_builders created for spxmod"
    )

    # Create and fit spxmod model
    logger.info(f"Fitting spxmod model with data size {df_train.shape}")
    model = XModel.from_config(xmodel_args)
    model.fit(data=df_train, data_span=df, **xmodel_fit_args)

    # Create prediction, residuals, and coefs
    logger.info("Calculating predictions and residuals")
    residual_calculator = ResidualCalculator(config.mtype)
    df[config.pred] = model.predict(df)
    residuals = residual_calculator.get_residual(
        df, config.pred, config.obs, config.weights
    )
    df_coef = get_coef(model)

    # Save results
    dataif.dump_spxmod(model, f"submodels/{submodel_id}/model.pkl")
    dataif.dump_spxmod(
        pd.concat([df, residuals], axis=1)[
            config.ids + ["residual", "residual_se", config.pred]
        ],
        f"submodels/{submodel_id}/predictions.parquet",
    )
    dataif.dump_spxmod(df_coef, f"submodels/{submodel_id}/coef.csv")


def main() -> None:
    fire.Fire(spxmod_model)
