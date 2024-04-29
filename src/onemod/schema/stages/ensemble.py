from typing import Literal

from onemod.schema.base import StageConfig


class EnsembleConfig(StageConfig):
    """Ensemble configuration class.

    Parameters
    ----------
    groupby
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list, which means all points are run in a single model.
    max_attempts
        Maximum number of attempts to run the Jobmon task associated
        with the stage. Default is 1.
    metric
        Metric used to compute model performance. Default is "rmse".
    score
        Score used to compute ensemble weights. Default is "rover".
    top_pct_score
        Models must be within top_pct_score of the best model to be
        included in the ensemble (i.e., score >= (1 - top_pct_score) *
        highest_score). Only used for the "rover" score function.
        Default is 1.0, which means all models are included in the
        ensemble.
    top_pct_model
        Percentage of highest scoring models to include in the ensemble.
        Only used for the "rover" score funciton. Default is 1.0, which
        means all models are included.
    psi
        Smoothing parameter for the "codem" score function. Default is
        1.0.

    Example
    -------
    All of the ensemble fields have default values equivalent to the
    following configuration.

    .. code-block:: yaml

        ensemble:
          groupby: []
          max_attempts: 1
          metric: rmse
          score: rover
          top_pct_score: 1.0
          top_pct_model: 1.0
          psi: 1.0

    """

    metric: Literal["rmse", "winsorized_rmse"] = "rmse"
    score: Literal["avg", "rover", "codem", "best"] = "rover"
    top_pct_score: float = 1.0
    top_pct_model: float = 1.0
    psi: float = 1.0
