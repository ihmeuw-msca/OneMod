"""Run ensemble model."""

import warnings
from pathlib import Path
from typing import Any, Optional

import fire
import numpy as np
import pandas as pd

from onemod.modeling.metric import Metric
from onemod.utils import Subsets, get_handle


def get_predictions(directory: str, holdout_id: Any, pred: str) -> pd.DataFrame:
    """Load available smoother predictions.

    Parameters
    ----------
    directory
        Path to the experiment directory.
    holdout_id
        Holdout ID for which predictions are requested.
    pred
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
    directory = Path(directory)

    dataif, _ = get_handle(directory)
    if holdout_id == "full":
        weave_file = "predictions.parquet"
    else:
        weave_file = f"predictions_{holdout_id}.parquet"

    try:
        df_smoother = dataif.load_weave(weave_file)
        # Use concat to add a level to the column multi-index
        df_smoother = pd.concat([df_smoother[pred]], axis=1, keys=["weave"])
    except FileNotFoundError:
        # No weave smoother results, initialize empty df
        warnings.warn("No weave predictions found for ensemble stage.")
        df_smoother = pd.DataFrame()

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
    subsets: Subsets,
    metric_name: str,
    obs: str,
    **kwargs,
) -> float:
    """Get smoother performance.

    Parameters
    ----------
    row : pd.Series
        Row containing the subset details.
    df_holdout : pd.DataFrame
        Dataframe containing the holdout data.
    subsets : Subsets
        Subsets object containing the subset data.
    metric_name : str
        Metric used to compute model performance.
    obs : str
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
    metric = Metric(metric_name)
    df_holdout = subsets.filter_subset(
        df_holdout[df_holdout[row["holdout_id"]] == 1], row["subset_id"]
    )
    return metric(df_holdout, obs, tuple(row[:3]), **kwargs)


def get_weights(
    df_performance: pd.DataFrame,
    subsets: Subsets,
    metric: str,
    score: str,
    top_pct_score: float = 1.0,
    top_pct_model: float = 1.0,
    psi: float = 1.0,
) -> pd.Series:
    """Get smoother weights.

    Parameters
    ----------
    df_performance : pd.DataFrame
        Dataframe containing smoother performance metrics.
    subsets : Subsets
        Subsets object containing the subset data.
    metric : str
        Metric used to compute model performance.
    score : str
        Score function used to compute ensemble weights.
    top_pct_score : float, optional
        Models must be within top_pct_score of the best model to be
        included in the ensemble (i.e., score >= (1 - top_pct_score) *
        highest_score). Only used for the "rover" score function.
        Default is 1.0, which means all models are included in the
        ensemble.
    top_pct_model : float, optional
        Percentage of highest scoring models to include in the ensemble.
        Only used for the "rover" score function. Default is 1.0, which
        means all models are included.
    psi : float, optional
        Smoothing parameter for "codem" score. Default is 1.0.

    Returns
    -------
    pd.Series
        Smoother weights.

    Raises
    ------
    ValueError
        If an invalid weight score is provided.

    """
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
    top_pct_score: float = 1.0,
    top_pct_model: float = 1.0,
    psi: float = 1.0,
) -> pd.Series:
    """Get subset weights.

    Parameters
    ----------
    performance : pd.Series
        Series containing performance metrics for each subset.
    score : str
        Score function used to compute ensemble weights.
    top_pct_score : float, optional
        Models must be within top_pct_score of the best model to be
        included in the ensemble (i.e., score >= (1 - top_pct_score) *
        highest_score). Only used for the "rover" score function.
        Default is 1.0, which means all models are included in the
        ensemble.
    top_pct_model : float, optional
        Percentage of highest scoring models to include in the ensemble.
        Only used for the "rover" score function. Default is 1.0, which
        means all models are included.
    psi : float, optional
        Smoothing parameter for "codem" score. Default is 1.0.

    Returns
    -------
    pd.Series
        Series containing the subset weights.

    Notes
    -----
    Available score functions are:
    - "avg": Assigns equal weight to all subsets.
    - "rover": Uses the score function from the ModRover package
    - "codem": Uses the score function from the CODEm model.
    - "best": Assigns all weight to the subset with the highest score.

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


def ensemble_model(directory: str, *args: Any, **kwargs: Any) -> None:
    """Run ensemble model on smoother predictions.

    Parameters
    ----------
    directory : str
        Path to the experiment directory.

    """
    dataif, config = get_handle(directory)

    stage_config = config.ensemble

    subsets = Subsets(
        "ensemble",
        stage_config,
        subsets=dataif.load_ensemble("subsets.csv"),
    )

    # Load input data and smoother predictions
    df_input = dataif.load_data()
    df_full = get_predictions(directory, "full", config.pred)

    # Get smoother out-of-sample performance by holdout set
    df_performance = pd.merge(
        left=df_full.columns.to_frame(index=False),
        right=pd.Series(config.holdouts, name="holdout_id"),
        how="cross",
    )
    df_performance = df_performance.merge(
        right=pd.Series(subsets.get_subset_ids(), name="subset_id"),
        how="cross",
    )
    df_list = []

    for holdout_id, df in df_performance.groupby("holdout_id"):
        predictions = get_predictions(directory, holdout_id, config.pred)
        predictions.columns = predictions.columns.to_flat_index()
        df_holdout = pd.merge(
            left=df_input[config.ids + [config.obs, holdout_id]],
            right=predictions,
            on=config.ids,
        )
        df[stage_config.metric] = df.apply(
            lambda row: get_performance(
                row,
                df_holdout,
                subsets,
                stage_config.metric,
                config.obs,
                **kwargs,
            ),
            axis=1,
        )
        df_list.append(df)

    # Get smoother weights
    columns = ["smoother_id", "model_id", "param_id", "subset_id"]
    metric_name = stage_config.metric
    groups = pd.concat(df_list).groupby(columns)
    mean = groups.mean(numeric_only=True).rename(
        {metric_name: f"{metric_name}_mean"}, axis=1
    )
    std = groups.std(numeric_only=True).rename(
        {metric_name: f"{metric_name}_std"}, axis=1
    )
    df_performance = pd.concat([mean, std], axis=1)
    df_performance["weight"] = get_weights(
        df_performance,
        subsets,
        stage_config.metric,
        stage_config.score,
        stage_config.top_pct_score,
        stage_config.top_pct_model,
    )
    dataif.dump_ensemble(df_performance, "performance.csv", index=True)

    # Get ensemble predictions
    df_list = []
    for subset_id in subsets.get_subset_ids():
        indices = [
            tuple(index)
            for index in subsets.filter_subset(df_input, subset_id)[
                config.ids
            ].values
        ]
        df_subset = df_full.loc[indices]
        df_list.append(
            (df_subset * df_performance["weight"][:, :, :, subset_id])
            .T.sum()
            .reset_index()
            .rename(columns={0: config.pred})
        )
    df_pred = pd.concat(df_list)

    dataif.dump_ensemble(df_pred, "predictions.parquet")


def main() -> None:
    """Main entry point of the module.

    This function uses the Fire library to allow the ensemble_model
    function to be invoked from the command line.

    """
    fire.Fire(ensemble_model)
