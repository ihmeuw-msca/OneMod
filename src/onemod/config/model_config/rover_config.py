"""ModRover covariate selection stage settings.

FIXME: improve top_pct_score and top_pct_learner descriptions

"""

from typing import Literal

from pydantic import Field, NonNegativeInt, model_validator
from typing_extensions import Self

from onemod.config import ModelConfig
from onemod.dtypes import UniqueList


class RoverConfig(ModelConfig):
    """ModRover covariate selection stage settings.

    For more details, please check out the ModRover package
    `documentation <https://ihmeuw-msca.github.io/modrover/>`_.

    Attributes
    ----------
    cov_exploring : UniqueList[str]
        Names of covariates to explore.
    cov_fixed : UniqueList[str], optional
        Fixed covariate names. Default is {'intercept'}.
    strategies : UniqueList[str], optional
        Set of strategies to use; either 'full', 'forward', and/or
        'backward'. Default is {'forward'}.
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

    cov_exploring: UniqueList[str]
    cov_fixed: UniqueList[str] = ["intercept"]
    strategies: UniqueList[Literal["full", "forward", "backward"]] = ["forward"]
    top_pct_score: float = Field(ge=0, le=1, default=0.1)
    top_pct_learner: float = Field(ge=0, le=1, default=1.0)
    t_threshold: float = Field(ge=0, default=1.0)
    min_covs: NonNegativeInt | None = None
    max_covs: NonNegativeInt | None = None

    # FIXME: Validate after pipeline settings passed to stage settings
    # @model_validator(mode="after")
    # def check_holdouts(self) -> Self:
    #     """Make sure holdouts present."""
    #     if self.holdout_columns is None:
    #         raise ValueError("Holdout columns required for rover stage")
    #     return self

    @model_validator(mode="after")
    def check_min_max(self) -> Self:
        """Make sure min_covs <= max_covs."""
        if self.min_covs is not None and self.max_covs is not None:
            if self.min_covs > self.max_covs:
                raise ValueError("min_covs > max_covs")
        return self
