"""Run regmod smooth model, currently the main goal of this step is to smooth
the covariate coefficients across age groups.
"""

from functools import partial
from typing import Callable

import fire
from loguru import logger
import numpy as np
import pandas as pd
from regmodsm.model import Model

from onemod.utils import get_handle


def get_residual_computation_function(
    model_type: str,
    col_obs: str,
    col_pred: str,
) -> Callable:
    """
    Calculate the residual for a given row based on the specified model type.

    Parameters:
        row (pd.Series): The row containing the observation and prediction data.
        model_type (str): Type of the statistical model (e.g., 'binomial', 'poisson', 'tobit').
        col_obs (str): Column name for the observed values.
        col_pred (str): Column name for the predicted values.

    Returns:
        float: The calculated residual value.

    Raises:
        ValueError: If the specified model_type is unsupported.
    """

    # TODO: can these be vectorized functions?
    callable_map = {
        "binomial": partial(
            lambda row, obs, pred: (row[obs] - row[pred])
            / (row[pred] * (1 - row[pred])),
            obs=col_obs,
            pred=col_pred,
        ),
        "poisson": partial(
            lambda row, obs, pred: row[obs] / row[pred] - 1, obs=col_obs, pred=col_pred
        ),
        "gaussian": partial(
            lambda row, obs, pred: row[obs] - row[pred], obs=col_obs, pred=col_pred
        ),
    }

    try:
        return callable_map[model_type]
    except KeyError:
        raise ValueError(f"Unsupported {model_type=}")


def get_residual_se_function(
    model_type: str,
    col_pred: str,
    col_weights: str,
) -> Callable:
    """
    Calculate the residual standard error for a given row based on the specified model type.

    Parameters:
        row (pd.Series): The row containing the observation and prediction data.
        model_type (str): Type of the statistical model (e.g., 'binomial', 'poisson', 'tobit').
        col_pred (str): Column name for the predicted values.
        col_weights (str): Column name for the weights.

    Returns:
        float: The calculated residual standard error value.

    Raises:
        ValueError: If the specified model_type is unsupported.
    """

    callable_map = {
        "binomial": partial(
            lambda row, pred, weights: 1
            / np.sqrt(row[weights] * row[pred] * (1 - row[pred])),
            pred=col_pred,
            weights=col_weights,
        ),
        "poisson": partial(
            lambda row, pred, weights: 1 / np.sqrt(row[weights] * row[pred]),
            pred=col_pred,
            weights=col_weights,
        ),
        "gaussian": partial(
            lambda row, weights: 1 / np.sqrt(row[weights]), weights=col_weights
        ),
    }

    try:
        return callable_map[model_type]
    except KeyError:
        raise ValueError(f"Unsupported {model_type=}")


def get_coef(model: Model) -> pd.DataFrame:
    """
    Get coefficient information from the specified model.

    Parameters:
        model (Model): The statistical model object containing coefficient data.

    Returns:
        pd.DataFrame: A DataFrame containing coefficient, dimension,
            and dimension value information.
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


def regmod_smooth_model(experiment_dir: str) -> None:
    """Run regmod smooth model smooth the age coefficients across different age
    groups.

    Parameters
    ----------
    experiment_dir
        Parent folder where the experiment is run.
        - ``experiment_dir / config / settings.yaml`` contains rover modeling settings
        - ``experiment_dir / results / rover`` stores all rover results

    Outputs
    -------
    model.pkl
        Regmodsm model instance for diagnostics.
    coef.csv
        Coefficients from different age groups.
    predictions.parquet
        Predictions with residual information.
    """
    dataif, global_config = get_handle(experiment_dir)

    regmod_smooth_config = global_config.regmod_smooth

    # Create regmod smooth parameters
    var_groups = regmod_smooth_config.model.var_groups
    coef_bounds = regmod_smooth_config.model.coef_bounds
    lam = regmod_smooth_config.model.lam

    var_group_keys = [
        (var_group["col"], var_group.get("dim")) for var_group in var_groups
    ]

    selected_covs = dataif.load_rover_covsel("selected_covs.yaml")
    if not selected_covs:
        selected_covs = []

    logger.info(f"Running smoothing with {selected_covs} as chosen covariates.")

    # add selected covariates as var_group with age_mid as the dimension
    for cov in selected_covs:
        if (cov, "age_mid") not in var_group_keys:
            var_groups.append(dict(col=cov, dim="age_mid"))

    logger.info(f"{len(var_groups)} var_groups created for smoothing.")

    # default settings for everyone
    for var_group in var_groups:
        cov = var_group["col"]
        if "uprior" not in var_group:
            var_group["uprior"] = tuple(
                map(float, coef_bounds.get(cov, [-np.inf, np.inf]))
            )
        if "lam" not in var_group:
            var_group["lam"] = lam

    # Create regmod smooth model
    model = Model(
        model_type=regmod_smooth_config.mtype,
        obs=global_config.col_obs,
        dims=regmod_smooth_config.model.dims,
        var_groups=var_groups,
        weights=regmod_smooth_config.model.weights,
    )

    df = dataif.load(global_config.input_path)
    df_train = df.query(
        f"({global_config.col_test} == 0) & {global_config.col_obs}.notnull()"
    )

    logger.info(f"Fitting the model with data size {df_train.shape}")

    # Fit regmod smooth model
    model.fit(df_train, data_dim_vals=df, **regmod_smooth_config.regmod_fit)
    # Create prediction and residuals
    logger.info("Model fit, calculating residuals")
    df[global_config.col_pred] = model.predict(df)
    residual_func = get_residual_computation_function(
        model_type=regmod_smooth_config.mtype,
        col_obs=global_config.col_obs,
        col_pred=global_config.col_pred,
    )

    residual_se_func = get_residual_se_function(
        model_type=regmod_smooth_config.mtype,
        col_pred=global_config.col_pred,
        col_weights=regmod_smooth_config.model.weights,
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
    dataif.dump_regmod_smooth(model, "model.pkl")
    dataif.dump_regmod_smooth(df_coef, "coef.csv")
    dataif.dump_regmod_smooth(df, "predictions.parquet")


def main() -> None:
    fire.Fire(regmod_smooth_model)
