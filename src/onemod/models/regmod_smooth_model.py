"""Run regmod smooth model, currently the main goal of this step is to smooth
the covariate coefficients across age groups.
"""
import fire
import numpy as np
import pandas as pd
from scipy.stats import norm
from regmodsm.model import Model
from onemod.utils import get_data_interface


def get_residual(
    row: pd.Series, model_type: str, col_obs: str, col_pred: str, inv_link: str
) -> float:
    """
    Calculate the residual for a given row based on the specified model type and inverse link function.

    Parameters:
        row (pd.Series): The row containing the observation and prediction data.
        model_type (str): Type of the statistical model (e.g., 'binomial', 'poisson', 'tobit').
        col_obs (str): Column name for the observed values.
        col_pred (str): Column name for the predicted values.
        inv_link (str): Inverse link function ('expit' for logistic, 'exp' for exponential, etc.).

    Returns:
        float: The calculated residual value.

    Raises:
        ValueError: If the specified model_type and inv_link pair is unsupported.
    """

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
    """
    Calculate the residual standard error for a given row based on the specified model type and inverse link function.

    Parameters:
        row (pd.Series): The row containing the observation and prediction data.
        model_type (str): Type of the statistical model (e.g., 'binomial', 'poisson', 'tobit').
        col_obs (str): Column name for the observed values.
        col_pred (str): Column name for the predicted values.
        inv_link (str): Inverse link function ('expit' for logistic, 'exp' for exponential, etc.).

    Returns:
        float: The calculated residual standard error value.

    Raises:
        ValueError: If the specified model_type and inv_link pair is unsupported.
    """
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


def get_coef(model: Model) -> pd.DataFrame:
    """
    Get coefficient information from the specified model.

    Parameters:
        model (Model): The statistical model object containing coefficient data.

    Returns:
        pd.DataFrame: A DataFrame containing coefficient, dimension, and dimension value information.
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


def regmod_smooth_model(experiment_dir: str, submodel_id: str) -> None:
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
    dataif = get_data_interface(experiment_dir)
    settings = dataif.load_settings()

    # Create regmod smooth parameters
    var_groups = settings["regmod_smooth"]["Model"].get("var_groups", [])
    coef_bounds = settings["regmod_smooth"]["Model"].get("coef_bounds", {})
    lam = settings["regmod_smooth"]["Model"].get("lam", 0.0)

    var_group_keys = [
        (var_group["col"], var_group.get("dim")) for var_group in var_groups
    ]

    selected_covs = dataif.load_rover_covsel("selected_covs.yaml")

    # add selected covariates as var_group with age_mid as the dimension
    base_settings = RegmodSmoothConfiguration(**settings["regmod_smooth"])


    for cov in selected_covs:
        if (cov, "age_mid") not in var_group_keys:
            var_groups.append(dict(col=cov, dim="age_mid"))

    # default settings for everyone
    for var_group in var_groups:
        cov = var_group["col"]
        if "uprior" not in var_group:
            var_group["uprior"] = tuple(map(float, coef_bounds.get(cov, [-100, 100])))
        if "lam" not in var_group:
            var_group["lam"] = lam

    # Create regmod smooth model
    model = Model(
        model_type=settings["regmod_smooth"]["Model"]["model_type"],
        obs=settings["regmod_smooth"]["Model"]["obs"],
        dims=settings["regmod_smooth"]["Model"]["dims"],
        var_groups=var_groups,
        weights=settings["regmod_smooth"]["Model"]["weights"],
    )

    # Slice the dataframe to only columns of interest
    expected_columns = [
        *settings["rover_covsel"]["Rover"]["cov_exploring"],
        *settings["col_id"],
        settings["col_obs"],
        settings["col_test"],
        settings["rover_covsel"]["Rover"]["weights"],
        *[col["name"] for col in settings["regmod_smooth"]["Model"]["dims"]],
    ]
    expected_columns = list(set(expected_columns))

    df = dataif.load(settings["input_path"], columns=expected_columns)
    df_train = df.query(
        f"({settings['col_test']} == 0) & {settings['col_obs']}.notnull()"
    )

    # Fit regmod smooth model
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

    df_coef = get_coef(model)

    # Save results
    dataif.dump_regmod_smooth(model, "model.pkl")
    dataif.dump_regmod_smooth(df_coef, "coef.csv")
    dataif.dump_regmod_smooth(df, "predictions.parquet")


def main() -> None:
    fire.Fire(regmod_smooth_model)
