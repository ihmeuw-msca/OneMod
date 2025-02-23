"""ModRover covariate selection stage settings.

FIXME: improve top_pct_score and top_pct_learner descriptions

"""

from typing import Literal

from pydantic import Field, NonNegativeInt, model_validator
from typing_extensions import Self

from onemod.config import Config, StageConfig
from onemod.dtypes import UniqueList


class RoverConfig(StageConfig):
    """ModRover covariate selection stage settings.

    For more details, please check out the ModRover package
    `documentation <https://ihmeuw-msca.github.io/modrover/>`_.

    Attributes `model_type`, `observation_column`, `weights_column`, and
    `holdout_columns` must be included in either the stage's config or
    the pipeline's config.

    Attributes
    ----------
    model_type : str, optional
        Model type; either 'binomial', 'gaussian', or 'poisson'. Default
        is None.
    observation_column : str, optional
        Observation column name for pipeline input. Default is None.
    weights_column : str, optional
        Weights column name for pipeline input. The weights column
        should contain nonnegative floats. Default is None.
    train_column : str, optional
        Training data column name. The train column should contain
        values 1 (train) or 0 (test). If no train column is provided,
        all non-null observations will be included in training. Default
        is None.
    holdout_columns : list of str or None, optional
        Holdout column names. The holdout columns should contain values
        1 (holdout), 0 (train), or NaN (missing observations). Holdout
        sets are used to evaluate stage model out-of-sample performance.
        Default is None.
    coef_bounds : dict, optional
        Dictionary of coefficient bounds with entries
        cov_name: (lower, upper). Default is None.
    cov_exploring : list of str
        Names of covariates to explore.
    cov_fixed : list of str, optional
        Fixed covariate names. Default is ['intercept'].
    cov_groupby : list of str, optional
        Column names used to create data subsets; covariates are
        selected separately for each data subset. Default is an empty
        list.
    strategies : list of str, optional
        Set of strategies to use; either 'full', 'forward', and/or
        'backward'. Default is ['forward'].
    top_pct_score : float in [0, 1], optional
        Percentage of learners with top scores to consider.
        Default is 0.1.
    top_pct_learner : float in [0, 1], optional
        Percentage of learners to consider. Default is 1.
    t_threshold : positive float, optional
        T-statistic threshold to consider as a covariate selection
        criterion. Default is 1.
    min_covs, max_covs : nonnegative int or None, optional
        Minimum/maximum number of covariates selected from
        cov_exploring, regardless of t_threshold value. Default is None.

    """

    model_type: Literal["binomial", "gaussian", "poisson"] | None = None
    observation_column: str | None = None
    weights_column: str | None = None
    train_column: str | None = None
    holdout_columns: UniqueList[str] | None = None
    coef_bounds: dict[str, tuple[float, float]] | None = None
    cov_exploring: UniqueList[str]
    cov_fixed: UniqueList[str] = ["intercept"]
    cov_groupby: UniqueList[str] = []
    strategies: UniqueList[Literal["full", "forward", "backward"]] = ["forward"]
    top_pct_score: float = Field(ge=0, le=1, default=0.1)
    top_pct_learner: float = Field(ge=0, le=1, default=1.0)
    t_threshold: float = Field(ge=0, default=1.0)
    min_covs: NonNegativeInt | None = None
    max_covs: NonNegativeInt | None = None
    _pipeline_config: Config = Config()
    _required: list[str] = [
        "model_type",
        "observation_column",
        "weights_column",
        "holdout_columns",
    ]

    @model_validator(mode="after")
    def check_min_max(self) -> Self:
        """Make sure min_covs <= max_covs."""
        if self.min_covs is not None and self.max_covs is not None:
            if self.min_covs > self.max_covs:
                raise ValueError("min_covs > max_covs")
        return self
