"""Collect onemod stage submodel results."""

from warnings import warn

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pplkit.data.interface import DataInterface

from onemod.schema import OneModConfig
from onemod.utils import (
    get_handle,
    get_rover_covsel_submodels,
    get_weave_submodels,
    parse_weave_submodel,
)


def _get_rover_covsel_summaries(dataif: DataInterface) -> pd.DataFrame:
    submodel_ids = get_rover_covsel_submodels(dataif.experiment)
    summaries = []
    for submodel_id in submodel_ids:
        summary = dataif.load_rover_covsel(
            f"submodels/{submodel_id}/summary.csv"
        )
        summary["submodel_id"] = submodel_id
        summaries.append(summary)
    summaries = pd.concat(summaries, axis=0)

    # Merge with the existing subsets
    subsets = dataif.load_rover_covsel("subsets.csv")
    subsets["submodel_id"] = [f"subset{i}" for i in subsets["subset_id"]]
    summaries = summaries.merge(
        subsets.drop("subset_id", axis=1), on="submodel_id", how="left"
    )
    summaries["abs_t_stat"] = summaries.eval("abs(coef / coef_sd)")
    return summaries


def _get_selected_covs(dataif: DataInterface) -> list[str]:
    summaries = _get_rover_covsel_summaries(dataif)
    t_threshold = dataif.load_settings()["rover_covsel"]["t_threshold"]

    selected_covs = (
        summaries.groupby("cov")["abs_t_stat"]
        .mean()
        .reset_index()
        .query(f"abs_t_stat >= {t_threshold}")["cov"]
        .tolist()
    )
    logger.info(f"Selected covariates: {selected_covs}")
    return selected_covs


def _plot_rover_covsel_results(
    dataif: DataInterface,
    summaries: pd.DataFrame,
    covs: list[str] | None = None,
) -> plt.Figure:
    """TODO: We hard-coded that the submodels for rover_covsel model are vary
    across age groups and use age mid as x axis of the plot.
    """

    logger.info("Plotting coefficient magnitudes by age.")
    config = OneModConfig(**dataif.load_settings())

    # add age_mid to summary
    df_age = dataif.load(
        config.input_path, columns=["age_group_id", "age_mid"]
    ).drop_duplicates()

    summaries = summaries.merge(df_age, on="age_group_id", how="left")
    df_covs = summaries.groupby("cov")
    covs = covs or list(df_covs.groups.keys())
    logger.info(
        f"Starting to plot for {len(covs)} groups of data of size {df_age.shape}"
    )

    fig, ax = plt.subplots(len(covs), 1, figsize=(8, 2 * len(covs)))
    ax = [ax] if len(covs) == 1 else ax
    for i, cov in enumerate(covs):
        df_cov = df_covs.get_group(cov).sort_values(by="age_mid")
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


def _plot_spxmod_results(
    dataif: DataInterface, summaries: pd.DataFrame
) -> plt.Figure | None:
    """TODO: same with _plot_rover_covsel_results"""
    selected_covs = dataif.load_rover_covsel("selected_covs.yaml")
    if not selected_covs:
        warn("There are no covariates selected, skip `plot_spxmod_results`")
        return None

    df_covs = dataif.load_spxmod("coef.csv").groupby("cov")

    fig = _plot_rover_covsel_results(dataif, summaries, covs=selected_covs)
    logger.info(
        f"Plotting smoothed covariates for {len(selected_covs)} covariates."
    )
    for ax, cov in zip(fig.axes, selected_covs):
        df_cov = df_covs.get_group(cov)
        ax.plot(
            df_cov["age_mid"],
            df_cov["coef"],
            fmt="o-",
            alpha=0.5,
            label="spxmod",
        )
        ax.legend(fontsize="xx-small")
    return fig


def collect_results_rover_covsel(directory: str) -> None:
    """Collect rover covariate selection results. Process all the significant
    covariates for each sub group. If a covaraite is significant across more
    than half of the subgroups if will be selected.

    This step will save ``selected_covs.yaml`` with a list of selected
    covariates in the rover results folder.
    """
    dataif, _ = get_handle(directory)

    selected_covs = _get_selected_covs(dataif)
    dataif.dump_rover_covsel(selected_covs, "selected_covs.yaml")

    # Concatenate summaries and save
    logger.info("Saving concatenated rover coefficient summaries.")
    summaries = _get_rover_covsel_summaries(dataif)
    dataif.dump_rover_covsel(summaries, "summaries.csv")

    fig = _plot_rover_covsel_results(dataif, summaries)
    fig.savefig(dataif.rover_covsel / "coef.pdf", bbox_inches="tight")


def collect_results_spxmod(directory: str) -> None:
    """This step is used for creating diagnostics."""
    dataif, _ = get_handle(directory)
    summaries = _get_rover_covsel_summaries(dataif)
    fig = _plot_spxmod_results(dataif, summaries)
    if fig is not None:
        fig.savefig(dataif.spxmod / "smooth_coef.pdf", bbox_inches="tight")


def collect_results_weave(directory: str) -> None:
    """Collect weave submodel results."""
    dataif, config = get_handle(directory)

    submodel_ids = get_weave_submodels(directory)
    for holdout_id in config.holdouts + ["full"]:
        df_list = []
        for submodel_id in submodel_ids:
            submodel = parse_weave_submodel(submodel_id)
            if submodel["holdout_id"] == holdout_id:
                df = dataif.load_weave(f"submodels/{submodel_id}.parquet")
                df["model_id"] = submodel["model_id"]
                df["param_id"] = str(submodel["param_id"])
                df_list.append(df)
        df_pred = pd.pivot(
            data=pd.concat(df_list, ignore_index=True),
            index=config.ids,
            columns=["model_id", "param_id"],
            values=["residual", "residual_se", config.pred],
        )
        if holdout_id == "full":
            dataif.dump_weave(df_pred, "predictions.parquet")
        else:
            dataif.dump_weave(df_pred, f"predictions_{holdout_id}.parquet")


def collect_results(stage_name: str, directory: str) -> None:
    callable_map = {
        "rover_covsel": collect_results_rover_covsel,
        "spxmod": collect_results_spxmod,
        "weave": collect_results_weave,
    }
    try:
        func = callable_map[stage_name]
    except KeyError:
        raise ValueError(f"Stage name {stage_name} is not valid.")

    func(directory)


def main() -> None:
    fire.Fire(collect_results)
