"""Run spxmod stage.

This stage fits a model with the covariates selected in the rover stage,
using priors or splines to smooth covariate coefficients across age
groups (based on the 'age_mid" column in the input data).

"""

from functools import partial
from typing import Callable

import fire
import numpy as np
import pandas as pd
from loguru import logger
from spxmod.model import XModel

from onemod.schema import OneModConfig
from onemod.utils import Subsets, get_handle


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
    """Calculate the residual SE for a given row based on model type.

    Parameters
    ----------
    model_type
        Model type (e.g., 'binomial', 'poisson', 'tobit').
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
    """Get coefficient information from the specified model.

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


def _build_xmodel_args(config: OneModConfig, selected_covs: list[str]) -> dict:
    """Format config data for spxmod xmodel.

    Model includes a coefficient for each of the selected covariates and
    age group (based on the 'age_mid' column in the input data).

    TODO: Update for spline variables.

    """
    xmodel_args = config.spxmod.xmodel.model_dump()
    coef_bounds = xmodel_args.pop("coef_bounds")
    lam = xmodel_args.pop("lam")

    space_keys = [space["name"] for space in xmodel_args["spaces"]]
    var_builder_keys = [
        (var_builder["name"], var_builder["space"])
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
    stage, using priors or splines to smooth covariate coefficients
    across age groups (based on the 'age_mid" column in the input data).

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

    # Load data and filter by subset
    subsets = Subsets(
        "spxmod", stage_config, subsets=dataif.load_spxmod("subsets.csv")
    )
    subset_id = int(submodel_id.removeprefix("subset"))
    df = subsets.filter_subset(dataif.load_data(), subset_id)
    df_train = df.query(f"({config.test} == 0) & {config.obs}.notnull()")

    # Load selected covs and filter by subset
    selected_covs = dataif.load_rover_covsel("selected_covs.yaml")
    for col_name in config.groupby:
        col_val = subsets.get_column(col_name, subset_id)
        selected_covs = selected_covs.query(f"{col_name} == {repr(col_val)}")
    selected_covs = selected_covs.tolist()
    logger.info(f"Running spxmod with selected_covariates: {selected_covs}")

    # Create spxmod parameters
    xmodel_args = _build_xmodel_args(config, selected_covs)
    xmodel_fit_args = stage_config.xmodel_fit
    logger.info(
        f"{len(xmodel_args['var_builders'])} var_builders created for spxmod"
    )

    # Create and fit spxmod model
    logger.info(f"Fitting spxmod model with data size {df_train.shape}")
    model = XModel.from_config(xmodel_args)
    model.fit(data=df_train, data_span=df, **xmodel_fit_args)

    # Create prediction and residuals
    logger.info("Calculating predictions and residuals")
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
    dataif.dump_spxmod(model, f"submodels/{submodel_id}/model.pkl")
    dataif.dump_spxmod(df_coef, f"submodels/{submodel_id}/coef.csv")
    dataif.dump_spxmod(
        df[config.ids + ["residual", "residual_se", config.pred]],
        f"submodels/{submodel_id}/predictions.parquet",
    )


def main() -> None:
    fire.Fire(spxmod_model)
