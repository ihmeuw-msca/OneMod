"""Collect onemod stage submodel results."""

from warnings import warn

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pplkit.data.interface import DataInterface
from scipy.special import logit

from onemod.schema.models.onemod_config import OneModConfig
from onemod.utils import (get_binom_adjusted_se, get_handle,
                          get_rover_covsel_submodels, get_swimr_submodels,
                          get_weave_submodels)


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
    settings = OneModConfig(**dataif.load_settings())

    # add age_mid to summary
    df_age = dataif.load(
        settings.input_path, columns=["age_group_id", "age_mid"]
    ).drop_duplicates()

    summaries = summaries.merge(df_age, on="age_group_id", how="left")
    df_covs = summaries.groupby("cov")
    covs = covs or list(df_covs.groups.keys())
    logger.info(
        f"Starting to plot for {len(covs)} groups of data of size {df_age.shape}"
    )

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
    if not selected_covs:
        warn(
            "There are no covariates selected, skip `plot_regmod_smooth_results`"
        )
        return None

    df_coef = (
        dataif.load_regmod_smooth("coef.csv")
        .query("dim == 'age_mid'")
        .rename(columns={"dim_val": "age_mid"})
    )
    df_covs = df_coef.groupby("cov")

    fig = _plot_rover_covsel_results(dataif, summaries, covs=selected_covs)
    logger.info(
        f"Plotting smoothed covariates for {len(selected_covs)} covariates."
    )
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
        ax.legend(fontsize="xx-small")
    return fig


def collect_results_rover_covsel(experiment_dir: str) -> None:
    """Collect rover covariate selection results. Process all the significant
    covariates for each sub group. If a covaraite is significant across more
    than half of the subgroups if will be selected.

    This step will save ``selected_covs.yaml`` with a list of selected
    covariates in the rover results folder.
    """
    dataif, _ = get_handle(experiment_dir)

    selected_covs = _get_selected_covs(dataif)
    dataif.dump_rover_covsel(selected_covs, "selected_covs.yaml")

    # Concatenate summaries and save
    logger.info("Saving concatenated rover coefficient summaries.")
    summaries = _get_rover_covsel_summaries(dataif)
    dataif.dump_rover_covsel(summaries, "summaries.csv")

    fig = _plot_rover_covsel_results(dataif, summaries)
    fig.savefig(dataif.rover_covsel / "coef.pdf", bbox_inches="tight")


def collect_results_regmod_smooth(experiment_dir: str) -> None:
    """This step is used for creating diagnostics."""
    dataif, _ = get_handle(experiment_dir)
    summaries = _get_rover_covsel_summaries(dataif)
    fig = _plot_regmod_smooth_results(dataif, summaries)
    if fig is not None:
        fig.savefig(
            dataif.regmod_smooth / "smooth_coef.pdf", bbox_inches="tight"
        )


def collect_results_swimr(experiment_dir: str) -> None:
    """Collect swimr submodel results."""
    dataif, settings = get_handle(experiment_dir)

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


def collect_results_weave(experiment_dir: str) -> None:
    """Collect weave submodel results."""
    dataif, config = get_handle(experiment_dir)

    # Collect submodel results
    submodel_ids = get_weave_submodels(experiment_dir)
    for holdout_id in config.col_holdout + ["full"]:
        df_pred = pd.concat(
            [
                dataif.load_weave(f"submodels/{submodel_id}.parquet").astype(
                    {"param_id": str}
                )
                for submodel_id in submodel_ids
                if submodel_id.split("_")[3] == holdout_id
            ],
            ignore_index=True,
        ).drop_duplicates()
        values = ["residual", "residual_var", config.col_pred]

        # Add binomial SE
        if holdout_id == "full" and config.mtype == "binomial":
            df_pred = df_pred.merge(
                right=dataif.load_regmod_smooth("predictions.parquet")[
                    config.col_id
                    + [
                        config.col_obs,
                        config.regmod_smooth.model.weights,
                        "regmod_logit_SE",
                    ]
                ],
                on=config.col_id,
            )
            df_pred["logitspace_pred"] = logit(df_pred[config.col_pred])
            df_pred["logitspace_adjusted_se"] = get_binom_adjusted_se(
                df_pred[config.col_pred],
                df_pred["residual_var"] + df_pred["regmod_logit_SE"] ** 2,
                df_pred[config.col_obs],
                df_pred[config.regmod_smooth.model.weights],
            )
            values += ["logitspace_pred", "logitspace_adjusted_se"]

        # Save weave results
        df_pred = pd.pivot(
            data=df_pred,
            index=config.col_id,
            columns=["model_id", "param_id"],
            values=values,
        )
        if holdout_id == "full":
            dataif.dump_weave(df_pred, "predictions.parquet")
        else:
            dataif.dump_weave(df_pred, f"predictions_{holdout_id}.parquet")


def collect_results(stage_name: str, experiment_dir: str) -> None:
    callable_map = {
        "rover_covsel": collect_results_rover_covsel,
        "regmod_smooth": collect_results_regmod_smooth,
        "swimr": collect_results_swimr,
        "weave": collect_results_weave,
    }
    try:
        func = callable_map[stage_name]
    except KeyError:
        raise ValueError(f"Stage name {stage_name} is not valid.")

    func(experiment_dir)


def main() -> None:
    fire.Fire(collect_results)
