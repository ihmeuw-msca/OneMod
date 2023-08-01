"""Run ensemble model."""
from pathlib import Path
from typing import Any, Optional, Union

import fire
import numpy as np
import pandas as pd

from onemod.utils import as_list, get_ensemble_input, load_settings, Subsets


def get_predictions(
    experiment_dir: Union[Path, str], holdout_id: Any, col_pred: str
) -> pd.DataFrame:
    """Load available smoother predictions.

    Parameters
    ----------
    experiment_dir : Union[Path, str]
        Path to the experiment directory.
    holdout_id : Any
        Holdout ID for which predictions are requested.
    col_pred : str
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
    if holdout_id == "full":
        swimr_path = Path(experiment_dir / "results" / "swimr" / "predictions.parquet")
        weave_path = Path(experiment_dir / "results" / "weave" / "predictions.parquet")
    else:
        swimr_path = Path(
            experiment_dir / "results" / "swimr" / f"predictions_{holdout_id}.parquet"
        )
        weave_path = Path(
            experiment_dir / "results" / "weave" / f"predictions_{holdout_id}.parquet"
        )
    if swimr_path.exists() and weave_path.exists():
        df_smoother = pd.merge(
            left=pd.concat(
                [pd.read_parquet(swimr_path)[col_pred]], axis=1, keys=["swimr"]
            ),
            right=pd.concat(
                [pd.read_parquet(weave_path)[col_pred]], axis=1, keys=["weave"]
            ),
            left_index=True,
            right_index=True,
        )
    elif swimr_path.exists():
        df_smoother = pd.concat(
            [pd.read_parquet(swimr_path)[col_pred]], axis=1, keys=["swimr"]
        )
    elif weave_path.exists():
        df_smoother = pd.concat(
            [pd.read_parquet(weave_path)[col_pred]], axis=1, keys=["weave"]
        )
    else:
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
    df_holdout = df_holdout[df_holdout[row["holdout_id"]] == 1]
    if subsets is not None:
        df_holdout = subsets.filter_subset(df_holdout, row["subset_id"])
    if metric == "rmse":
        return np.sqrt(np.mean((df_holdout[col_obs] - df_holdout[tuple(row[:3])]) ** 2))
    raise ValueError(f"Invalid performance metric: {metric}")


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
            df_performance.loc[metric + "_mean"],
            score,
            top_pct_score,
            top_pct_model,
            psi,
        )
    performance = df_performance.T
    for subset_id in subsets.get_subset_ids():
        subset_idx = pd.IndexSlice[:, :, :, subset_id]
        performance.loc[subset_idx, "weight"] = get_subset_weights(
            performance.loc[subset_idx, metric + "_mean"],
            score,
            top_pct_score,
            top_pct_model,
            psi,
        )
    return performance.T.loc["weight"]


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


def ensemble_model(experiment_dir: Union[Path, str], *args: Any) -> None:
    """Run ensemble model on smoother predictions.

    Parameters
    ----------
    experiment_dir : Union[Path, str]
        Path to the experiment directory.

    """
    experiment_dir = Path(experiment_dir)
    ensemble_dir = experiment_dir / "results" / "ensemble"
    settings = load_settings(experiment_dir / "config" / "settings.yml")
    subsets = Subsets(
        "ensemble",
        settings["ensemble"],
        subsets=pd.read_csv(ensemble_dir / "subsets.csv"),
    )

    # Load input data and smoother predictions
    df_input = get_ensemble_input(settings)
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
    for holdout_id, df in df_performance.groupby("holdout_id"):
        df_holdout = pd.merge(
            left=df_input,
            right=get_predictions(experiment_dir, holdout_id, settings["col_pred"]),
            on=settings["col_id"],
        )
        df[settings["ensemble"]["metric"]] = df.apply(
            lambda row: get_performance(
                row,
                df_holdout,
                subsets,
                settings["ensemble"]["metric"],
                settings["col_obs"],
            ),
            axis=1,
        )
        df_list.append(df)

    # Get smoother weights
    columns = ["smoother_id", "model_id", "param_id"]
    if "groupby" in settings["ensemble"]:
        columns += ["subset_id"]
    df_performance = pd.pivot_table(
        pd.concat(df_list),
        columns=columns,
        values=[settings["ensemble"]["metric"]],
        aggfunc=[np.mean, np.std],
    )
    df_performance.index = df_performance.index.droplevel()
    df_performance.rename(
        index={
            "mean": settings["ensemble"]["metric"] + "_mean",
            "std": settings["ensemble"]["metric"] + "_std",
        },
        inplace=True,
    )
    if settings["ensemble"]["score"] == "avg":
        df_performance.loc["weight"] = get_weights(
            df_performance,
            subsets,
            settings["ensemble"]["metric"],
            settings["ensemble"]["score"],
        )
    else:
        df_performance.loc["weight"] = get_weights(
            df_performance,
            subsets,
            settings["ensemble"]["metric"],
            settings["ensemble"]["score"],
            settings["ensemble"]["top_pct_score"],
            settings["ensemble"]["top_pct_model"],
        )
    df_performance.T.to_csv(ensemble_dir / "performance.csv")

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
                (df_subset * df_performance.loc["weight"][:, :, :, subset_id])
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
    df_pred.to_parquet(ensemble_dir / "predictions.parquet")


def main() -> None:
    """Main entry point of the module.

    This function uses the Fire library to allow the rover_model function to be
    invoked from the command line.
    """
    fire.Fire(ensemble_model)
