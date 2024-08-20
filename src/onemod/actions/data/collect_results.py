"""Collect onemod stage submodel results."""

from warnings import warn

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pplkit.data.interface import DataInterface

from onemod.schema import OneModConfig
from onemod.utils import get_handle, get_submodels, parse_weave_submodel


def _get_rover_covsel_summaries(dataif: DataInterface) -> pd.DataFrame:
    subsets = dataif.load_rover_covsel("subsets.csv")

    # Collect coefficient summaries
    summaries = []
    for subset_id in subsets["subset_id"]:
        summary = dataif.load_rover_covsel(
            f"submodels/subset{subset_id}/summary.csv"
        )
        summary["subset_id"] = subset_id
        summaries.append(summary)
    summaries = pd.concat(summaries)

    # Merge with existing subsets and add statistic
    summaries = summaries.merge(subsets, on="subset_id", how="left")
    summaries["abs_t_stat"] = summaries.eval("abs(coef / coef_sd)")
    return summaries


def _get_selected_covs(
    summaries: pd.DataFrame,
    groupby: list[str],
    t_threshold: float,
    min_covs: float | None = None,
    max_covs: float | None = None,
) -> pd.DataFrame:
    """Select covariates from cov_exploring.
    Parameters
    ----------
    summaries : pandas.DataFrame
        Covariate summaries across rover models.
    groupby : list[str]
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc.
    t_threshold : float
        T-statistic threshold to consider as a covariate selection
        criterion.
    min_covs, max_covs : float or None, optional
        Minimum/maximum number of covariates to select from
        cov_exploring, regardless of t_threshold value. Default is None.
    Returns
    -------
    pandas.DataFrame
        Selected covariates and their t-statistics.
    """
    selected_covs = []
    logger.info("Selecting covariates")
    for group, df in summaries.groupby(groupby):
        t_stats = (
            df.groupby(groupby + ["cov"])["abs_t_stat"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .eval(f"selected = abs_t_stat >= {t_threshold}")
        )
        if min_covs is not None and t_stats["selected"].sum() < min_covs:
            t_stats.loc[: min_covs - 1, "selected"] = True
        if max_covs is not None and t_stats["selected"].sum() > max_covs:
            t_stats.loc[max_covs:, "selected"] = False
        selected = t_stats.query("selected").drop(columns="selected")
        selected_covs.append(selected)
        logger.info(
            ", ".join(
                [
                    f"{id_name}: {id_val}"
                    for id_name, id_val in zip(groupby, group)
                ]
                + [f"covs: {selected["cov"].values}"]
            )
        )
    return pd.concat(selected_covs)


def _plot_rover_covsel_results(
    dataif: DataInterface,
    summaries: pd.DataFrame,
    covs: list[str] | None = None,
) -> plt.Figure:
    """TODO: We hard-coded that the submodels for rover_covsel model are vary
    across age groups and use age mid as x axis of the plot.
    """
    # TODO: Plot by submodel?
    logger.info("Plotting coefficient magnitudes by age.")

    # add age_mid to summary
    df_age = dataif.load_data(
        columns=["age_group_id", "age_mid"]
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
    # TODO: Plot by submodel?
    selected_covs = (
        dataif.load_rover_covsel("selected_covs.csv")["cov"].unique().tolist()
    )
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
            df_cov["age_mid"], df_cov["coef"], "o-", alpha=0.5, label="spxmod"
        )
        ax.legend(fontsize="xx-small")
    return fig


def collect_results_rover_covsel(directory: str) -> None:
    """Collect rover covariate selection results.

    Collect covariate summaries from submodels, select covariates based
    on t-statistic and save as ``selected_covs.csv``, and plot
    covariate coefficients by age group.

    """
    dataif, config = get_handle(directory)

    # Concatenate summaries and save
    logger.info("Saving concatenated rover coefficient summaries.")
    summaries = _get_rover_covsel_summaries(dataif)
    dataif.dump_rover_covsel(summaries, "summaries.csv")

    # Select covariates and save
    selected_covs = _get_selected_covs(
        summaries, config.groupby, config.rover_covsel.t_threshold
    )
    dataif.dump_rover_covsel(selected_covs, "selected_covs.csv")

    # Plot coefficients and save
    fig = _plot_rover_covsel_results(dataif, summaries)
    fig.savefig(dataif.rover_covsel / "coef.pdf", bbox_inches="tight")


def collect_results_spxmod(directory: str) -> None:
    """This step is used for creating diagnostics."""
    dataif, _ = get_handle(directory)

    # Collect submodel predictions
    predictions = []
    subsets = dataif.load_spxmod("subsets.csv")
    for subset_id in subsets["subset_id"]:
        df = dataif.load_spxmod(
            f"submodels/subset{subset_id}/predictions.parquet"
        )
        predictions.append(df)
    dataif.dump_spxmod(pd.concat(predictions), "predictions.parquet")

    # Collect submodel coefficients
    coef = []
    for subset_id in subsets["subset_id"]:
        df = dataif.load_spxmod(f"submodels/subset{subset_id}/coef.csv")
        df["subset_id"] = subset_id
        coef.append(df)
    coef = pd.merge(
        left=pd.concat(coef).reset_index(drop=True),
        right=subsets.drop("stage_id", axis=1),
        on="subset_id",
        how="left",
    )
    dataif.dump_spxmod(coef, "coef.csv")

    # Plot coefficients
    summaries = _get_rover_covsel_summaries(dataif)
    fig = _plot_spxmod_results(dataif, summaries)
    if fig is not None:
        fig.savefig(dataif.spxmod / "smooth_coef.pdf", bbox_inches="tight")


def collect_results_weave(directory: str) -> None:
    """Collect weave submodel results."""
    dataif, config = get_handle(directory)

    submodel_ids = get_submodels("weave", directory)
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
