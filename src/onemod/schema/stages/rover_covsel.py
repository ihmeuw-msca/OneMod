from onemod.schema.base import Config, StageConfig


class RoverInit(Config):
    """Rover class initialization arguments. To create the rover model it will
    need addition configuration `mtype`, `obs`, `weights` from the global
    config. For more details please check
    `modrover <https://ihmeuw-msca.github.io/modrover/>`_.

    Parameters
    ----------
    cov_fixed
        List of fixed covariates. Default is a list with covariate intercept.
    cov_exploring
        List of covariates to explore. Default is an empty list.

    """

    cov_fixed: list[str] = ["intercept"]
    cov_exploring: list[str] = []


class RoverFit(Config):
    """Rover fit function arguments. For more details please check
    `modrover <https://ihmeuw-msca.github.io/modrover/>`_.

    Parameters
    ----------
    strategies
        List of strategies to use. Default is `["forward"]`.
    top_pct_score
        Percentage of models with top scores to consider. Default is 0.1.
    top_pct_learner
        Percentage of models to consider. Default is 1.0.
    coef_bounds
        Dictionary of coefficient bounds. Default is an empty dictionary.

    """

    strategies: list[str] = ["forward"]
    top_pct_score: float = 0.1
    top_pct_learner: float = 1.0
    coef_bounds: dict[str, tuple[float, float]] = {}


class RoverCovselConfig(StageConfig):
    """Rover covariate selection stage configurations.

    Parameters
    ----------
    rover
        Rover class initialization arguments.
    rover_fit
        Rover fit function arguments.
    t_threshold
        T-statistic threshold to consider as a covariate selection criterion.
        Default is 1.0.

    Example
    -------
    All of the fields have default values. And it is equivalent to the following
    configuration for the ensemble section.

    .. code-block:: yaml

        rover_covsel:
          groupby: []
          max_attempts: 1
          max_batch: -1
          rover:
            cov_fixed: ["intercept"]
            cov_exploring: []
          rover_fit:
            strategies: ["forward"]
            top_pct_score: 0.1
            top_pct_learner: 1.0
            coef_bounds: {}
          t_threshold: 1.0

    """

    rover: RoverInit = RoverInit()
    rover_fit: RoverFit = RoverFit()
    t_threshold: float = 1.0