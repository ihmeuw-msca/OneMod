"""Collect onemod stage submodel results."""
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
from pplkit.data.interface import DataInterface

import fire
import pandas as pd

from onemod.utils import (
    as_list,
    get_rover_covsel_submodels,
    get_swimr_submodels,
    get_weave_submodels,
    load_settings,
)


def _get_rover_covsel_summaries(dataif: DataInterface) -> pd.DataFrame:
    submodel_ids = get_rover_covsel_submodels(dataif.experiment)
    summaries = []
    for submodel_id in submodel_ids:
        summary = dataif.load_rover(f"covsel/submodels/{submodel_id}/summary.csv")
        summary["submodel_id"] = submodel_id
        summaries.append(summary)
    summaries = pd.concat(summaries, axis=0)
    return summaries


def _get_selected_covs(dataif: DataInterface) -> list[str]:
    summaries = _get_rover_covsel_summaries(dataif)
    selected_covs = (
        summaries.groupby("cov")["significant"]
        .mean()
        .reset_index()
        .query("significant >= 0.5")["cov"]
        .tolist()
    )
    return selected_covs


def _plot_rover_covsel_results(
    dataif: DataInterface, covs: list[str] | None = None
) -> plt.Figure:
    """TODO: We hard-coded that the submodels for rover_covsel model are vary
    across age groups and use age mid as x axis of the plot.
    """
    summaries = _get_rover_covsel_summaries(dataif)
    subsets = dataif.load_rover("covsel/subsets.csv")
    settings = dataif.load_experiment("config/settings.yml")

    # add age_mid to summary
    subsets["submodel_id"] = [f"subset{i}" for i in subsets["subset_id"]]
    summaries = summaries.merge(
        subsets[["submodel_id", "age_group_id"]], on="submodel_id", how="left"
    )
    df_age = dataif.load(
        settings["input_path"], columns=["age_group_id", "age_mid"]
    ).drop_duplicates()
    summaries = summaries.merge(df_age, on="age_group_id", how="left")

    df_covs = summaries.groupby("cov")
    covs = covs or list(df_covs.groups.keys())
    fig, ax = plt.subplots(len(covs), 1, figsize=(8, 2 * len(covs)))
    for i, cov in enumerate(covs):
        df_cov = df_covs.get_group(cov)
        ax[i].errorbar(
            df_cov["age_mid"],
            df_cov["coef"],
            yerr=1.96 * df_cov["coef_sd"],
            fmt="o-",
            alpha=0.5,
            label="rover_covsel",
        )
        ax[i].set_ylabel(cov)
        ax[i].axhline(0.0, linestyle="--")
    return fig


def _plot_regmod_smooth_results(dataif: DataInterface) -> plt.Figure:
    """TODO: same with _plot_rover_covsel_results"""
    selected_covs = dataif.load_rover("selected_covs.yaml")
    df_coef = (
        dataif.load_smooth("coef.csv")
        .query("dim == 'age_mid'")
        .rename(columns={"dim_val": "age_mid"})
    )
    df_covs = df_coef.groupby("cov")

    fig = _plot_rover_covsel_results(dataif, covs=selected_covs)
    for ax, cov in zip(fig.axes, selected_covs):
        df_cov = df_covs.get_group(cov)
        ax.errorbar(
            df_cov["age_mid"],
            df_cov["coef"],
            yerr=1.96 * df_cov["coef_sd"],
            fmt="o-",
            alpha=0.5,
            label="regmod_smooth",
        )
    return fig


def collect_rover_covsel_results(experiment_dir: Path | str) -> None:
    """Collect rover covariate selection results. Process all the significant
    covariates for each sub group. If a covaraite is significant across more
    than half of the subgroups if will be selected.

    This step will save ``selected_covs.yaml`` with a list of selected
    covariates in the rover results folder.
    """
    dataif = DataInterface(experiment=experiment_dir)
    dataif.add_dir("rover", dataif.experiment / "results" / "rover")

    selected_covs = _get_selected_covs(dataif)
    dataif.dump_rover(selected_covs, "selected_covs.yaml")

    fig = _plot_rover_covsel_results(dataif)
    fig.savefig(dataif.rover / "coef.pdf", bbox_inches="tight")


def collect_regmod_smooth_results(experiment_dir: Path | str) -> None:
    """This step is used for creating diagnostics."""
    dataif = DataInterface(experiment=experiment_dir)
    dataif.add_dir("rover", dataif.experiment / "results" / "rover")
    dataif.add_dir("smooth", dataif.rover / "smooth")
    fig = _plot_regmod_smooth_results(dataif)
    fig.savefig(dataif.rover / "smooth_coef.pdf", bbox_inches="tight")


def collect_swimr_results(experiment_dir: Union[Path, str]) -> None:
    """Collect swimr submodel results."""
    experiment_dir = Path(experiment_dir)
    swimr_dir = experiment_dir / "results" / "swimr"
    settings = load_settings(experiment_dir / "config" / "settings.yml")
    submodel_ids = get_swimr_submodels(experiment_dir)
    for holdout_id in as_list(settings["col_holdout"]) + ["full"]:
        df_list = []
        for submodel_id in [
            submodel_id
            for submodel_id in submodel_ids
            if submodel_id.split("_")[3] == holdout_id
        ]:
            df_list.append(
                pd.read_parquet(
                    swimr_dir / "submodels" / submodel_id / "predictions.parquet"
                ).astype({"param_id": str})
            )
        df_pred = pd.pivot(
            data=pd.concat(df_list, ignore_index=True),
            index=settings["col_id"],
            columns=["model_id", "param_id"],
            values=["residual", settings["col_pred"]],
        )
        if holdout_id == "full":
            df_pred.to_parquet(swimr_dir / "predictions.parquet")
        else:
            df_pred.to_parquet(swimr_dir / f"predictions_{holdout_id}.parquet")


def collect_weave_results(experiment_dir: Union[Path, str]) -> None:
    """Collect weave submodel results."""
    experiment_dir = Path(experiment_dir)
    weave_dir = experiment_dir / "results" / "weave"
    settings = load_settings(experiment_dir / "config" / "settings.yml")
    submodel_ids = get_weave_submodels(experiment_dir)
    for holdout_id in as_list(settings["col_holdout"]) + ["full"]:
        df_pred = pd.concat(
            [
                pd.read_parquet(
                    weave_dir / "submodels" / f"{submodel_id}.parquet"
                ).astype({"param_id": str})
                for submodel_id in submodel_ids
                if submodel_id.split("_")[3] == holdout_id
            ],
            ignore_index=True,
        )
        df_pred = pd.pivot(
            data=df_pred,
            index=settings["col_id"],
            columns=["model_id", "param_id"],
            values=["residual", settings["col_pred"]],
        )
        if holdout_id == "full":
            df_pred.to_parquet(weave_dir / "predictions.parquet")
        else:
            df_pred.to_parquet(weave_dir / f"predictions_{holdout_id}.parquet")


def main() -> None:
    fire.Fire(
        {
            "rover_covsel": collect_rover_covsel_results,
            "regmod_smooth": collect_regmod_smooth_results,
            "swimr": collect_swimr_results,
            "weave": collect_weave_results,
        }
    )
