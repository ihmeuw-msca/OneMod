"""Run ensemble model."""
from functools import reduce
from pathlib import Path
from typing import Any, Optional
import warnings

import fire
import numpy as np
import pandas as pd

from onemod.modeling.metric import Metric
from onemod.utils import as_list, get_data_interface, get_ensemble_input, Subsets


def get_predictions(
    experiment_dir: str, holdout_id: Any, col_pred: str
) -> pd.DataFrame:
    """Load available smoother predictions.

    Parameters
    ----------
    experiment_dir
        Path to the experiment directory.
    holdout_id
        Holdout ID for which predictions are requested.
    col_pred
        Column name for the prediction values.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the available smoother predictions.

    Raises
    ------
    FileNotFoundError
        If smoother results do not exist.

    """
    holdout_id = str(holdout_id)
    experiment_dir = Path(experiment_dir)

    dataif, _ = get_handle(experiment_dir)
    if holdout_id == "full":
        swimr_file = "predictions.parquet"
        weave_file = "predictions.parquet"
    else:
        swimr_file = f"predictions_{holdout_id}.parquet"
        weave_file = f"predictions_{holdout_id}.parquet"

    try:
        df_smoother = dataif.load_weave(weave_file)
        # Use concat to add a level to the column multi-index
        df_smoother = pd.concat([df_smoother[col_pred]], axis=1, keys="weave")
    except FileNotFoundError:
        # No weave smoother results, initialize empty df
        warnings.warn("No weave predictions found for ensemble stage.")
        df_smoother = pd.DataFrame()

    try:
        swimr_df = dataif.load_swimr(swimr_file)
        swimr_df = pd.concat([swimr_df[col_pred]], axis=1, keys="swimr")
        if df_smoother.empty:
            df_smoother = swimr_df
        else:
            df_smoother = pd.merge(
                df_smoother, swimr_df, left_index=True, right_index=True
            )
    except FileNotFoundError:
        warnings.warn("No swimr predictions found for ensemble stage.")

    if df_smoother.empty:
        raise FileNotFoundError("Smoother results do not exist")
    # df_smoother is always a multi-indexed dataframe, so df_smoother.columns is always a
    # MultiIndex object instead of an Index object.
    # If the dataframe is single-indexed the below line will raise an error.
    df_smoother.columns.rename("smoother_id", level=0, inplace=True)  # type: ignore
    return df_smoother


def get_performance(
    row: pd.Series,
    df_holdout: pd.DataFrame,
    subsets: Optional[Subsets],
    metric: str,
    col_obs: str,
    **kwargs,
) -> float:
    """Get smoother performance.

    Parameters
    ----------
    row : pd.Series
        Row containing the subset details.
    df_holdout : pd.DataFrame
        Dataframe containing the holdout data.
    subsets : Optional[Subsets]
        Subsets object containing the subset data.
    metric : str
        Performance metric to be used (e.g., "rmse").
    col_obs : str
        Column name for the observed values.

    Returns
    -------
    float
        Smoother performance metric.

    Raises
    ------
    ValueError
        If an invalid performance metric is provided.

    """
    df_holdout = df_holdout[
        (df_holdout[row["holdout_id"]] == 1)
        & (df_holdout["param_id"] == row["param_id"])
    ]
    if subsets is not None:
        df_holdout = subsets.filter_subset(df_holdout, row["subset_id"])

    metric = Metric(metric)
    return metric(df_holdout, col_obs, row.model_id, **kwargs)


def get_weights(
    df_performance: pd.DataFrame,
    subsets: Optional[Subsets],
    metric: str,
    score: str,
    top_pct_score: float = 0.0,
    top_pct_model: float = 0.0,
    psi: float = 0.0,
) -> pd.Series:
    """Get smoother weights.

    Parameters
    ----------
    df_performance : pd.DataFrame
        Dataframe containing smoother performance metrics.
    subsets : Optional[Subsets]
        Subsets object containing the subset data.
    metric : str
        Performance metric to be used for weighting (e.g., "rmse_mean").
    score : str
        Weighting score to be used (e.g., "avg", "rover", "codem", "best").
    top_pct_score : float, optional
        Percentage of top scores to be used for weighting, by default 0.
    top_pct_model : float, optional
        Percentage of top models to be used for weighting, by default 0.
    psi : float, optional
        Smoothing parameter for codem score, by default 0.

    Returns
    -------
    pd.Series
        Smoother weights.

    Raises
    ------
    ValueError
        If an invalid weight score is provided.

    """
    if subsets is None:
        return get_subset_weights(
            df_performance[metric + "_mean"],
            score,
            top_pct_score,
            top_pct_model,
            psi,
        )
    for subset_id in subsets.get_subset_ids():
        subset_idx = pd.IndexSlice[:, :, :, subset_id]
        df_performance.loc[subset_idx, "weight"] = get_subset_weights(
            df_performance.loc[subset_idx, metric + "_mean"],
            score,
            top_pct_score,
            top_pct_model,
            psi,
        )
    return df_performance["weight"]


def get_subset_weights(
    performance: pd.Series,
    score: str,
    top_pct_score: float = 0.0,
    top_pct_model: float = 0.0,
    psi: float = 0.0,
) -> pd.Series:
    """Get subset weights.

    This function calculates subset weights based on the given scoring method.

    Parameters
    ----------
    performance : pd.Series
        Series containing performance scores for each subset.
    score : str
        Scoring method to be used for computing weights. Available options are:
            - "avg": Assigns equal weights to all subsets.
            - "rover": Uses the ROVER (Rank-Ordered Vote Evaluation Results) method.
            - "codem": Uses the CODEm (Consistent Over Different Experiments) method.
            - "best": Assigns all weight to the subset with the best performance score.
    top_pct_score : float, optional
        The percentage of top-performing scores to be used for weighting (used only for
        "rover" score), by default 0.0.
    top_pct_model : float, optional
        The percentage of top models to be used for weighting (used only for "rover" score),
        by default 0.0.
    psi : float, optional
        Smoothing parameter for CODEm score (used only for "codem" score), by default 0.0.

    Returns
    -------
    pd.Series
        Series containing the subset weights.

    Raises
    ------
    ValueError
        If an invalid scoring method is provided.

    """
    if score == "avg":
        avg_scores = performance.copy()
        avg_scores[:] = 1
        return avg_scores / avg_scores.sum()
    if score == "rover":
        rover_scores = np.exp(-performance)
        argsort = np.argsort(rover_scores)[::-1]
        indices = rover_scores >= rover_scores[argsort[0]] * (1 - top_pct_score)
        num_learners = int(np.floor(len(rover_scores)) * top_pct_model) + 1
        indices[argsort[num_learners:]] = False
        rover_scores[~indices] = 0.0
        return rover_scores / rover_scores.sum()
    if score == "codem":
        ranks = np.argsort(np.argsort(performance))
        codem_scores = psi ** (len(performance) - ranks)
        return codem_scores / codem_scores.sum()
    if score == "best":
        best_scores = performance.copy()
        argsort = np.argsort(best_scores)
        best_scores[argsort[0]] = 1.0
        best_scores[argsort[1:]] = 0.0
        return best_scores
    raise ValueError(f"Invalid weight score: {score}")


def ensemble_model(experiment_dir: str, *args: Any, **kwargs: Any) -> None:
    """Run ensemble model on smoother predictions.

    Parameters
    ----------
    experiment_dir : Union[Path, str]
        Path to the experiment directory.

    """
    experiment_dir = Path(experiment_dir)
    dataif, _ = get_handle(experiment_dir)
    settings = dataif.load_settings()
    subsets_df = dataif.load_ensemble("subsets.csv")

    subsets = Subsets(
        "ensemble",
        settings["ensemble"],
        subsets=subsets_df,
    )

    # Load input data and smoother predictions
    df_input = get_ensemble_input(interface, settings)
    df_full = get_predictions(experiment_dir, "full", settings["col_pred"])

    # Get smoother out-of-sample performance by holdout set
    df_performance = pd.merge(
        left=df_full.columns.to_frame(index=False),
        right=pd.Series(as_list(settings["col_holdout"]), name="holdout_id"),
        how="cross",
    )
    if "groupby" in settings["ensemble"]:
        df_performance = df_performance.merge(
            right=pd.Series(subsets.get_subset_ids(), name="subset_id"),
            how="cross",
        )
    df_list = []
    id_cols = settings["col_id"]

    for holdout_id, df in df_performance.groupby("holdout_id"):
        predictions = get_predictions(experiment_dir, holdout_id, settings["col_pred"])
        # Iteratively merge on prediction columns
        df_holdout = reduce(
            lambda df, smoother: pd.merge(
                left=df,
                right=predictions.loc[:, smoother].stack().reset_index(),
                how="right",
                on=id_cols + ["param_id"],
            ),
            predictions.columns.get_level_values("smoother_id"),
            pd.DataFrame(columns=id_cols + ["param_id"]),
        )
        df_holdout = pd.merge(df_holdout, df_input, on=id_cols)

        # Select holdout ids, and calc rmse by subset
        df[settings["ensemble"]["metric"]] = df.apply(
            lambda row: get_performance(
                row,
                df_holdout,
                subsets,
                settings["ensemble"]["metric"],
                settings["col_obs"],
                **kwargs,
            ),
            axis=1,
        )
        df_list.append(df)

    # Get smoother weights
    columns = ["smoother_id", "model_id", "param_id"]
    if "groupby" in settings["ensemble"]:
        columns += ["subset_id"]

    metric = settings["ensemble"]["metric"]
    full_df = pd.concat(df_list)
    mean = (
        full_df.groupby(columns)
        .mean(numeric_only=True)
        .rename({metric: f"{metric}_mean"}, axis=1)
    )
    std = (
        full_df.groupby(columns)
        .std(numeric_only=True)
        .rename({metric: f"{metric}_std"}, axis=1)
    )
    df_performance = pd.concat([mean, std], axis=1)
    if settings["ensemble"]["score"] == "avg":
        df_performance["weight"] = get_weights(
            df_performance,
            subsets,
            settings["ensemble"]["metric"],
            settings["ensemble"]["score"],
        )
    else:
        df_performance["weight"] = get_weights(
            df_performance,
            subsets,
            metric,
            settings["ensemble"]["score"],
            settings["ensemble"]["top_pct_score"],
            settings["ensemble"]["top_pct_model"],
        )
    dataif.dump_ensemble(df_performance.T, "performance.csv")

    # Get ensemble predictions
    if "groupby" in settings["ensemble"]:
        df_list = []
        for subset_id in subsets.get_subset_ids():
            indices = [
                tuple(index)
                for index in subsets.filter_subset(df_input, subset_id)[
                    as_list(settings["col_id"])
                ].values
            ]
            df_subset = df_full.loc[indices]
            df_list.append(
                (df_subset * df_performance["weight"][:, :, :, subset_id])
                .T.sum()
                .reset_index()
                .rename(columns={0: settings["col_pred"]})
            )
        df_pred = pd.concat(df_list)
    else:
        df_pred = (
            (df_full * df_performance.loc["weight"])
            .T.sum()
            .reset_index()
            .rename(columns={0: settings["col_pred"]})
        )
    dataif.dump_ensemble(df_pred, "predictions.parquet")


def main() -> None:
    """Main entry point of the module.

    This function uses the Fire library to allow the rover_model function to be
    invoked from the command line.
    """
    fire.Fire(ensemble_model)
