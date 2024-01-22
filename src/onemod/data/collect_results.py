"""Collect onemod stage submodel results."""
import matplotlib.pyplot as plt
from warnings import warn
from pplkit.data.interface import DataInterface

import fire
from loguru import logger
import pandas as pd

from onemod.schema.models.parent_config import ParentConfiguration
from onemod.utils import (
    as_list,
    get_rover_covsel_submodels,
    get_swimr_submodels,
    get_weave_submodels,
    get_data_interface,
)


def _get_rover_covsel_summaries(dataif: DataInterface) -> pd.DataFrame:
    submodel_ids = get_rover_covsel_submodels(dataif.experiment)
    summaries = []
    for submodel_id in submodel_ids:
        summary = dataif.load_rover_covsel(f"submodels/{submodel_id}/summary.csv")
        summary["submodel_id"] = submodel_id
        summaries.append(summary)
    summaries = pd.concat(summaries, axis=0)

    # Merge with the existing subsets
    subsets = dataif.load_rover_covsel("subsets.csv")
    subsets["submodel_id"] = [f"subset{i}" for i in subsets["subset_id"]]
    summaries = summaries.merge(
        subsets.drop("subset_id", axis=1), on="submodel_id", how="left"
    )
    return summaries


def _get_selected_covs(dataif: DataInterface) -> list[str]:
    summaries = _get_rover_covsel_summaries(dataif)

    summaries['abs_t_stat']=(summaries['coef']/summaries['coef_sd']).abs()
    t_threshold = dataif.load_settings().get('rover_t_threshold', 2)
    #TODO also add fixed feature count option


    selected_t_covs = (
        summaries.groupby("cov")["abs_t_stat"]
        .mean()
        .reset_index()
        .query(f"abs_t_stat >= {t_threshold}")["cov"]
        .tolist()
    )

    logger.info(f"Selected covariates: {selected_t_covs}")
    return selected_t_covs


def _plot_rover_covsel_results(
    dataif: DataInterface, summaries: pd.DataFrame, covs: list[str] | None = None
) -> plt.Figure:
    """TODO: We hard-coded that the submodels for rover_covsel model are vary
    across age groups and use age mid as x axis of the plot.
    """

    logger.info("Plotting coefficient magnitudes by age.")
    settings = ParentConfiguration(**dataif.load_settings())

    # add age_mid to summary
    df_age = dataif.load(
        settings.input_path, columns=["age_group_id", "age_mid"]
    ).drop_duplicates()

    summaries = summaries.merge(df_age, on="age_group_id", how="left")
    df_covs = summaries.groupby("cov")
    covs = covs or list(df_covs.groups.keys())
    logger.info(f"Starting to plot for {len(covs)} groups of data of size {df_age.shape}")


    fig, ax = plt.subplots(len(covs), 1, figsize=(8, 2 * len(covs)))
    for i, cov in enumerate(covs):
        df_cov = df_covs.get_group(cov)
        if i % 5 == 0:
            logger.info(f"Plotting for group {i}")
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

    logger.info("Completed plotting of rover results.")
    return fig


def _plot_regmod_smooth_results(
    dataif: DataInterface, summaries: pd.DataFrame
) -> plt.Figure | None:
    """TODO: same with _plot_rover_covsel_results"""
    selected_covs = dataif.load_rover_covsel("selected_covs.yaml")
    if len(selected_covs) == 0:
        warn("There are no covariates selected, skip `plot_regmod_smooth_results`")
        return None

    df_coef = (
        dataif.load_regmod_smooth("coef.csv")
        .query("dim == 'age_mid'")
        .rename(columns={"dim_val": "age_mid"})
    )
    df_covs = df_coef.groupby("cov")

    fig = _plot_rover_covsel_results(dataif, summaries, covs=selected_covs)
    logger.info(f"Plotting smoothed covariates for {len(selected_covs)} covariates.")
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
        ax.legend(fontsize = 'xx-small')
    return fig


def collect_rover_covsel_results(experiment_dir: str) -> None:
    """Collect rover covariate selection results. Process all the significant
    covariates for each sub group. If a covaraite is significant across more
    than half of the subgroups if will be selected.

    This step will save ``selected_covs.yaml`` with a list of selected
    covariates in the rover results folder.
    """
    dataif = get_data_interface(experiment_dir)

    selected_covs = _get_selected_covs(dataif)
    dataif.dump_rover_covsel(selected_covs, "selected_covs.yaml")

    # Concatenate summaries and save
    logger.info("Saving concatenated rover coefficient summaries.")
    summaries = _get_rover_covsel_summaries(dataif)
    dataif.dump_rover_covsel(summaries, "summaries.csv")

    fig = _plot_rover_covsel_results(dataif, summaries)
    fig.savefig(dataif.rover_covsel / "coef.pdf", bbox_inches="tight")


def collect_regmod_smooth_results(experiment_dir: str) -> None:
    """This step is used for creating diagnostics."""
    dataif = get_data_interface(experiment_dir)
    summaries = _get_rover_covsel_summaries(dataif)
    fig = _plot_regmod_smooth_results(dataif, summaries)
    if fig is not None:
        fig.savefig(dataif.regmod_smooth / "smooth_coef.pdf", bbox_inches="tight")


def collect_swimr_results(experiment_dir: str) -> None:
    """Collect swimr submodel results."""
    dataif = get_data_interface(experiment_dir)
    settings = ParentConfiguration(**dataif.load_settings())

    submodel_ids = get_swimr_submodels(experiment_dir)
    for holdout_id in settings.col_holdout + ["full"]:
        df_list = []
        for submodel_id in [
            submodel_id
            for submodel_id in submodel_ids
            if submodel_id.split("_")[3] == holdout_id
        ]:
            df_list.append(
                dataif.load_swimr(
                    f"submodels/{submodel_id}/predictions.parquet"
                ).astype({"param_id": str})
            )
        df_pred = pd.pivot(
            data=pd.concat(df_list, ignore_index=True),
            index=settings.col_id,
            columns=["model_id", "param_id"],
            values=["residual", settings.col_pred],
        )
        if holdout_id == "full":
            dataif.dump_swimr(df_pred, "predictions.parquet")
        else:
            dataif.dump_swimr(df_pred, f"predictions_{holdout_id}.parquet")


def collect_weave_results(experiment_dir: str) -> None:
    """Collect weave submodel results."""
    dataif = get_data_interface(experiment_dir)
    settings = ParentConfiguration(**dataif.load_settings())

    submodel_ids = get_weave_submodels(experiment_dir)
    for holdout_id in settings.col_holdout + ["full"]:
        df_pred = pd.concat(
            [
                dataif.load_weave(f"submodels/{submodel_id}.parquet").astype(
                    {"param_id": str}
                )
                for submodel_id in submodel_ids
                if submodel_id.split("_")[3] == holdout_id
            ],
            ignore_index=True,
        )
        df_pred = pd.pivot(
            data=df_pred,
            index=settings["col_id"],
            columns=["model_id", "param_id"],
            values=["residual", settings.col_pred],
        )
        if holdout_id == "full":
            dataif.dump_weave(df_pred, "predictions.parquet")
        else:
            dataif.dump_weave(df_pred, f"predictions_{holdout_id}.parquet")


def main() -> None:
    fire.Fire(
        {
            "rover_covsel": collect_rover_covsel_results,
            "regmod_smooth": collect_regmod_smooth_results,
            "swimr": collect_swimr_results,
            "weave": collect_weave_results,
        }
    )
