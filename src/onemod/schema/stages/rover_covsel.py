from onemod.schema.base import Config, StageConfig


class RoverInit(Config):
    """Rover class initialization arguments."""

    cov_fixed: list[str] = ["intercept"]
    cov_exploring: list[str] = []


class RoverFit(Config):
    """Rover fit function arguments."""

    strategies: list[str] = ["forward"]
    top_pct_score: float = 0.1
    top_pct_learner: float = 1.0
    coef_bounds: dict[str, tuple[float, float]] = {}


class RoverCovselConfig(StageConfig):
    """Rover covariate selection stage configurations."""

    rover: RoverInit = RoverInit()
    rover_fit: RoverFit = RoverFit()
    t_threshold: float = 1.0
