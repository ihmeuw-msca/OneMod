"""Run regmod smooth model, currently the main goal of this step is to smooth
the covariate coefficients across age groups.
"""
import fire
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from regmodsm.model import Model
from pplkit.data.interface import DataInterface


def get_residual(
    row: pd.Series, model_type: str, col_obs: str, col_pred: str, inv_link: str
) -> float:
    """Get residual."""
    # TODO: can these be vectorized functions?
    if model_type == "binomial" and inv_link == "expit":
        return (row[col_obs] - row[col_pred]) / (row[col_pred] * (1 - row[col_pred]))
    if model_type == "poisson" and inv_link == "exp":
        return row[col_obs] / row[col_pred] - 1
    if model_type == "tobit" and inv_link == "exp":
        if row[col_obs] > 0:
            return row[col_obs] / row[col_pred] - 1
        w = row[col_pred] / row["sigma"]
        term = w * np.imag(norm.logcdf(-w + 1e-6j)) / (1e-6)
        return -1 / (1 - w**2 + term)
    raise ValueError("Unsupported model_type and inv_link pair")


def get_residual_se(
    row: pd.Series, model_type: str, col_obs: str, col_pred: str, inv_link: str
) -> float:
    """Get residual standard error."""
    if model_type == "binomial" and inv_link == "expit":
        return 1 / np.sqrt(row[col_pred] * (1 - row[col_pred]))
    if model_type == "poisson" and inv_link == "exp":
        return 1 / np.sqrt(row[col_pred])
    if model_type == "tobit" and inv_link == "exp":
        if row[col_obs] > 0:
            return row["sigma"] / row[col_pred]
        w = row[col_pred] / row["sigma"]
        term = w * np.imag(norm.logcdf(-w + 1e-6j)) / (1e-6)
        return np.sqrt(1 / (term * (1 - w**2 + term)))
    raise ValueError("Unsupported model_type and inv_link pair")


def regmod_smooth_model(experiment_dir: Path | str, submodel_id: str) -> None:
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
    predictions.parquet
        Predictions with residual information.
    """
    dataif = DataInterface(experiment=experiment_dir)
    dataif.add_dir("config", dataif.experiment / "config")
    dataif.add_dir("rover", dataif.experiment / "results" / "rover_covsel")
    dataif.add_dir("smooth", dataif.experiment / "results" / "regmod_smooth")

    settings = dataif.load_config("settings.yml")

    # Create regmod smooth parameters
    var_groups = settings["regmod_smooth"]["Model"]["var_groups"]
    coef_bounds = settings["regmod_smooth"]["Model"]["coef_bounds"]

    selected_covs = dataif.load_rover("selected_covs.yaml")

    # Fill in default box constraint for selected covariates if not already provided
    for cov in selected_covs:
        if cov not in coef_bounds:
            coef_bounds[cov] = [-100, 100]

    for cov in selected_covs:
        var_group = dict(col=cov, dim="age_mid")
        if cov in coef_bounds:
            var_group.update(dict(uprior=tuple(map(float, coef_bounds[cov]))))
            # Optionally set smoothing parameter, defaults to 0 if not provided
            if "lambda" in settings["regmod_smooth"]["Model"]:
                var_group["lam"] = settings["regmod_smooth"]["Model"]["lambda"]

        var_groups.append(var_group)

    # Create regmod smooth model
    model = Model(
        model_type=settings["regmod_smooth"]["Model"]["model_type"],
        obs=settings["regmod_smooth"]["Model"]["obs"],
        dims=settings["regmod_smooth"]["Model"]["dims"],
        var_groups=var_groups,
        weights=settings["regmod_smooth"]["Model"]["weights"],
    )

    # Fit regmod smooth model
    df = dataif.load(settings["input_path"])
    df_train = df.query(f"{settings['col_test']} == 0")

    # Slice the dataframe to only columns of interest
    expected_columns = settings["rover_covsel"]["Rover"]["cov_exploring"]
    expected_columns.append(settings["col_obs"])
    for col in settings["regmod_smooth"]["Model"]["dims"]:
        expected_columns.append(col["name"])

    df_train = df_train[expected_columns]
    df_train = df_train[~(df_train[settings["col_obs"]].isnull())]

    model.fit(df_train, **settings["regmod_smooth"]["Model.fit"])

    # Create prediction and residuals
    df[settings["col_pred"]] = model.predict(df)
    df["residual"] = df.apply(
        lambda row: get_residual(
            row,
            settings["rover_covsel"]["Rover"]["model_type"],
            settings["col_obs"],
            settings["col_pred"],
            settings["rover_covsel"]["inv_link"],
        ),
        axis=1,
    )
    df["residual_se"] = df.apply(
        lambda row: get_residual_se(
            row,
            settings["rover_covsel"]["Rover"]["model_type"],
            settings["col_obs"],
            settings["col_pred"],
            settings["rover_covsel"]["inv_link"],
        ),
        axis=1,
    )

    # Save results
    dataif.dump_smooth(model, "model.pkl")
    dataif.dump_smooth(df, "predictions.parquet")


def main() -> None:
    fire.Fire(regmod_smooth_model)
