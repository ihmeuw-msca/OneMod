"""Run spxmod model, currently the main goal of this step is to smooth
the covariate coefficients across age groups.
"""

from functools import partial
from typing import Callable

import fire
import numpy as np
import pandas as pd
from loguru import logger
from spxmod.model import XModel

from onemod.schema.stages.spxmod import XModelInit
from onemod.utils import get_handle


def get_residual_computation_function(
    model_type: str, obs: str, pred: str
) -> Callable:
    """
    Calculate the residual for a given row based on the specified model type.

    Parameters
    ----------
    model_type
        Type of the statistical model (e.g., 'binomial', 'poisson', 'tobit').
    obs
        Column name for the observed values.
    pred
        Column name for the predicted values.

    Returns
    -------
    float
        The calculated residual value.

    Raises
    ------
    ValueError
        If the specified model_type is unsupported.

    """

    # TODO: can these be vectorized functions?
    callable_map = {
        "binomial": partial(
            lambda row, obs, pred: (row[obs] - row[pred])
            / (row[pred] * (1 - row[pred])),
            obs=obs,
            pred=pred,
        ),
        "poisson": partial(
            lambda row, obs, pred: row[obs] / row[pred] - 1, obs=obs, pred=pred
        ),
        "gaussian": partial(
            lambda row, obs, pred: row[obs] - row[pred], obs=obs, pred=pred
        ),
    }

    try:
        return callable_map[model_type]
    except KeyError:
        raise ValueError(f"Unsupported {model_type=}")


def get_residual_se_function(
    model_type: str, pred: str, weights: str
) -> Callable:
    """
    Calculate the residual standard error for a given row based on the specified model type.

    Parameters
    ----------
    model_type
        Type of the statistical model (e.g., 'binomial', 'poisson', 'tobit').
    pred
        Column name for the predicted values.
    weights
        Column name for the weights.

    Returns
    -------
    float
        The calculated residual standard error value.

    Raises
    ------
    ValueError
        If the specified model_type is unsupported.

    """

    callable_map = {
        "binomial": partial(
            lambda row, pred, weights: 1
            / np.sqrt(row[weights] * row[pred] * (1 - row[pred])),
            pred=pred,
            weights=weights,
        ),
        "poisson": partial(
            lambda row, pred, weights: 1 / np.sqrt(row[weights] * row[pred]),
            pred=pred,
            weights=weights,
        ),
        "gaussian": partial(
            lambda row, weights: 1 / np.sqrt(row[weights]), weights=weights
        ),
    }

    try:
        return callable_map[model_type]
    except KeyError:
        raise ValueError(f"Unsupported {model_type=}")


def get_coef(model: XModel) -> pd.DataFrame:
    """
    Get coefficient information from the specified model.

    Parameters
    ----------
    model
        The statistical model object containing coefficient data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing coefficient, dimension, and dimension value information.

    """
    df_coef = []
    for var_group in model.var_groups:
        dim = var_group.dim
        if dim is None:
            dim_vals = [np.nan]
            dim_name = "None"
        else:
            dim_vals = dim.vals
            dim_name = dim.name
        df_sub = pd.DataFrame(
            {
                "cov": var_group.col,
                "dim": dim_name,
                "dim_val": dim_vals,
            }
        )
        df_coef.append(df_sub)
    df_coef = pd.concat(df_coef, axis=0, ignore_index=True)
    df_coef["coef"] = model._model.opt_coefs
    df_coef["coef_sd"] = np.sqrt(np.diag(model._model.opt_vcov))
    return df_coef


def _build_xmodel_args(
    xmodel_config: XModelInit, selected_covs: list[str]
) -> dict:
    xmodel_args = xmodel_config.model_dump()
    coef_bounds = xmodel_args.pop("coef_bounds")
    lam = xmodel_args.pop("lam")

    space_keys = [space["name"] for space in xmodel_args["spaces"]]
    var_builder_keys = [
        (var_builder["name"], var_builder["space"]["name"])
        for var_builder in xmodel_args["var_builders"]
    ]

    # add age_mid if not specified in spaces
    if "age_mid" not in space_keys:
        xmodel_args["spaces"].append(
            dict(name="age_mid", dims=[dict(name="age_mid", type="numerical")])
        )

    for cov in selected_covs:
        if (cov, "age_mid") not in var_builder_keys:
            xmodel_args["var_builders"].append(dict(name=cov, space="age_mid"))

    # default settings for everyone
    for var_builder in xmodel_args["var_builders"]:
        cov = var_builder["name"]
        if "uprior" not in var_builder:
            var_builder["uprior"] = tuple(
                map(float, coef_bounds.get(cov, [-np.inf, np.inf]))
            )
        if "lam" not in var_builder:
            var_builder["lam"] = lam

    return xmodel_args


def spxmod_model(directory: str) -> None:
    """Run spxmod model smooth the age coefficients across different age
    groups.

    Parameters
    ----------
    directory
        Parent folder where the experiment is run.
        - ``directory / config / settings.yaml`` contains rover modeling settings
        - ``directory / results / spxmod`` stores all rover results

    Outputs
    -------
    model.pkl
        Regmodsm model instance for diagnostics.
    coef.csv
        Coefficients from different age groups.
    predictions.parquet
        Predictions with residual information.

    """
    dataif, config = get_handle(directory)
    stage_config = config.spxmod

    # load selected covs from previous stage
    selected_covs = dataif.load_rover_covsel("selected_covs.yaml")
    if not selected_covs:
        selected_covs = []

    logger.info(f"Running smoothing with {selected_covs} as chosen covariates.")

    # Create spxmod parameters
    xmodel_args = _build_xmodel_args(stage_config.xmodel, selected_covs)
    xmodel_fit_args = stage_config.xmodel_fit

    logger.info(
        f"{len(xmodel_args["var_builders"])} var_builders created for smoothing."
    )

    # Create spxmod model
    model = XModel(**xmodel_args)

    df = dataif.load_data()
    df_train = df.query(f"({config.test} == 0) & {config.obs}.notnull()")

    logger.info(f"Fitting the model with data size {df_train.shape}")

    # Fit spxmod model
    model.fit(data=df_train, data_span=df, **xmodel_fit_args)
    # Create prediction and residuals
    logger.info("XModel fit, calculating residuals")
    df[config.pred] = model.predict(df)
    residual_func = get_residual_computation_function(
        model_type=config.mtype,
        obs=config.obs,
        pred=config.pred,
    )
    residual_se_func = get_residual_se_function(
        model_type=config.mtype,
        pred=config.pred,
        weights=config.weights,
    )
    df["residual"] = df.apply(
        residual_func,
        axis=1,
    )
    df["residual_se"] = df.apply(
        residual_se_func,
        axis=1,
    )

    df_coef = get_coef(model)

    # Save results
    dataif.dump_spxmod(model, "model.pkl")
    dataif.dump_spxmod(df_coef, "coef.csv")
    dataif.dump_spxmod(df, "predictions.parquet")


def main() -> None:
    fire.Fire(spxmod_model)
