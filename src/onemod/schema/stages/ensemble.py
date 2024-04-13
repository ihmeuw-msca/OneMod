from onemod.schema.base import StageConfig


class EnsembleConfig(StageConfig):
    """Configuration class for ensemble stage. All fields have default values.

    Parameters
    ----------
    metric
        Metric to use for model evaluation. Default is "rmse".
    score
        Score to compute the model ensemble weights. Default is "neg_exp".
    top_pct_score
        Percentage of models have the top score to consider for ensemble
        weights. Default is 1.0, which means all models are considered.
    top_pct_model
        Percentage of models to consider for ensemble weights. Default is 1.0,
        which means all models are considered.

    Example
    -------
    All of the fields have default values. And it is equivalent to the following
    configuration for the ensemble section.

    .. code-block:: yaml

        ensemble:
          groupby: []
          max_attempts: 1
          max_batch: -1
          metric: rmse
          score: neg_exp
          top_pct_score: 1.0
          top_pct_model: 1.0

    """

    metric: str = "rmse"
    score: str = "neg_exp"
    top_pct_score: float = 1.0
    top_pct_model: float = 1.0
