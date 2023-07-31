"""Run regmod smooth model, currently the main goal of this step is to smooth
the covariate coefficients across age groups.
"""
import fire
import pandas as pd
from pathlib import Path
from regmodsm.model import Model
from pplkit.data.interface import DataInterface


def _get_selected_covs(experiment_dir: Path | str) -> list[str]:
    """Process all the significant covariates for each sub group. If a covaraite
    is significant across more than half of the subgroups if will be selected.
    """
    dataif = DataInterface(experiment=experiment_dir)
    dataif.add_dir("config", dataif.experiment / "config")
    dataif.add_dir("covsel", dataif.experiment / "results" / "rover" / "covsel")

    subsets = dataif.load_covsel("subsets.csv")
    summaries = []
    for subset_id in subsets.subset_id:
        submodel_id = f"subset{subset_id}"
        summary = dataif.load_covsel(f"{submodel_id}/summary.csv")
        summary["submodel_id"] = submodel_id
        summaries.append(summary)
    summaries = pd.concat(summaries, axis=0)

    selected_covs = (
        summaries.groupby("cov")["significant"]
        .mean()
        .reset_index()
        .query("significant >= 0.5")["cov"]
        .tolist()
    )

    return selected_covs


def get_residual(row: pd.Series, model_type: str, col_obs: str, inv_link: str) -> float:
    """Get residual."""
    if model_type == "binomial" and inv_link == "expit":
        return (row[col_obs] - row["p"]) / (row["p"] * (1 - row["p"]))
    if model_type == "poisson" and inv_link == "exp":
        return row[col_obs] / row["lam"] - 1
    if model_type == "tobit" and inv_link == "exp":
        if row[col_obs] > 0:
            return row[col_obs] / row["mu"] - 1
        w = row["mu"] / row["sigma"]
        term = w * np.imag(norm.logcdf(-w + 1e-6j)) / (1e-6)
        return -1 / (1 - w**2 + term)
    raise ValueError("Unsupported model_type and inv_link pair")


def get_residual_se(
    row: pd.Series, model_type: str, col_obs: str, inv_link: str
) -> float:
    """Get residual standard error."""
    if model_type == "binomial" and inv_link == "expit":
        return 1 / np.sqrt(row["p"] * (1 - row["p"]))
    if model_type == "poisson" and inv_link == "exp":
        return 1 / np.sqrt(row["lam"])
    if model_type == "tobit" and inv_link == "exp":
        if row[col_obs] > 0:
            return row["sigma"] / row["mu"]
        w = row["mu"] / row["sigma"]
        term = w * np.imag(norm.logcdf(-w + 1e-6j)) / (1e-6)
        return np.sqrt(1 / (term * (1 - w**2 + term)))
    raise ValueError("Unsupported model_type and inv_link pair")


def regmod_smooth_model(experiment_dir: Path | str) -> None:
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
    dataif.add_dir("covsel", dataif.experiment / "results" / "rover" / "covsel")
    dataif.add_dir("smooth", dataif.experiment / "results" / "rover" / "smooth")
    settings = dataif.load_config("settings.yml")

    # Create regmod smooth parameters
    var_groups = settings["regmod_smooth"]["Model"]["var_groups"]
    coef_bounds = settings["regmod_smooth"]["Model"]["coef_bounds"]

    selected_covs = _get_selected_covs(experiment_dir)
    for cov in selected_covs:
        var_group = dict(col=cov, dim="age_mid")
        if cov in coef_bounds:
            var_group.update(dict(uprior=tuple(map(float, coef_bounds[cov]))))
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
    df = dataif.load(settings["intput_path"])
    df_train = df.query("col_test == 0")
    model.fit(df_train, **settings["regmod_smooth"]["Model.fit"])

    # Create prediction and residuals
    df[settings["col_pred"]] = model.predict(df)
    df["residual"] = df.apply(
        lambda row: get_residual(
            row,
            settings["rover"]["model_type"],
            settings["col_obs"],
            settings["rover"]["inv_link"],
        ),
        axis=1,
    )
    df["residual_se"] = df.apply(
        lambda row: get_residual_se(
            row,
            settings["rover"]["model_type"],
            settings["col_obs"],
            settings["rover"]["inv_link"],
        ),
        axis=1,
    )
    df.drop(columns=settings["col_obs"], inplace=True)

    # Save results
    dataif.dump_smooth(model, "model.pkl")
    dataif.dump_smooth(df, "predictions.parquet")


def main() -> None:
    fire.Fire(regmod_smooth_model)
