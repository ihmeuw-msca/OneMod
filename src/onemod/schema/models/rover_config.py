from pydantic import BaseModel, Field

from onemod.schema.models.base import ParametrizedBaseModel


class RoverInit(BaseModel):
    cov_fixed: list[str] = []
    cov_exploring: list[str] = []
    weights: str


class RoverFitArgs(BaseModel):
    strategies: list[str] = ["forward"]
    top_pct_score: float = 0.1
    top_pct_learner: float = 1.0
    coef_bounds: tuple[float, float] | None = None


class RoverConfiguration(ParametrizedBaseModel):
    groupby: list[str] = []
    model_type: str = ""
    max_attempts: int | None = None
    Rover: RoverInit

    fit_args: RoverFitArgs = Field(default_factory=RoverFitArgs)

    def inherit(self) -> None:
        super().inherit(keys=["model_type", "groupby", "max_attempts", "max_batch"])
