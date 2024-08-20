"""Collect onemod stage submodel results."""

import warnings

import fire
import pandas as pd
from loguru import logger
from pplkit.data.interface import DataInterface

from onemod.diagnostics import plot_rover_covsel_results, plot_spxmod_results
from onemod.utils import get_handle, get_submodels, parse_weave_submodel


def _get_rover_covsel_summaries(dataif: DataInterface) -> pd.DataFrame:
    subsets = dataif.load_rover_covsel("subsets.csv")

    # Collect coefficient summaries
    summaries = []
    for subset_id in subsets["subset_id"]:
        try:
            summary = dataif.load_rover_covsel(
                f"submodels/subset{subset_id}/summary.csv"
            )
            summary["subset_id"] = subset_id
            summaries.append(summary)
        except FileNotFoundError:
            warnings.warn(
                f"rover_covsel subset {subset_id} missing summary.csv"
            )
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



def collect_results_rover_covsel(directory: str) -> None:
    """Collect rover covariate selection results.

    Collect covariate summaries from submodels, select covariates based
    on t-statistic and save as ``selected_covs.csv``, and plot
    covariate coefficients by age group.

    TODO: separate plots by groupings other than sex_id

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
    summaries = summaries.merge(
        dataif.load_data(columns=["age_group_id", "age_mid"]).drop_duplicates(),
        on="age_group_id",
        how="left",
    )
    if config.plots:
        if "sex_id" in config.groupby:
            for sex_id, df in summaries.groupby("sex_id"):
                fig = plot_rover_covsel_results(df)
                fig.savefig(
                    dataif.rover_covsel / f"coef_{sex_id}.pdf",
                    bbox_inches="tight",
                )
        else:
            fig = plot_rover_covsel_results(summaries)
            fig.savefig(dataif.rover_covsel / "coef.pdf", bbox_inches="tight")


def collect_results_spxmod(directory: str) -> None:
    """This step is used for creating diagnostics.

    FIXME: assumes rover stage has been run

    """
    dataif, config = get_handle(directory)

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
    if config.plots:
        summaries = pd.merge(
            left=dataif.load_rover_covsel("summaries.csv"),
            right=dataif.load_data(
                columns=["age_group_id", "age_mid"]
            ).drop_duplicates(),
            how="left",
        )
        for subset_id, df in coef.groupby("subset_id"):
            if "sex_id" in config.groupby:
                sex_id = df["sex_id"].unique()[0]
                fig = plot_spxmod_results(
                    summaries.query("sex_id == @sex_id"), df
                )
            else:
                fig = plot_spxmod_results(summaries, df)
        fig.savefig(
            dataif.spxmod / f"smooth_coef_{subset_id}.pdf", bbox_inches="tight"
        )


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
