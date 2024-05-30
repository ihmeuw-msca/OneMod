from onemod.schema.base import Config, StageConfig


class RoverInit(Config):
    """Rover class initialization arguments.

    To create a rover model, addition configuration args `mtype`, `obs`,
    and `weights` are taken from the OneMod config. For more details
    please check out the ModRover package
    `documentation <https://ihmeuw-msca.github.io/modrover/>`_.

    Parameters
    ----------
    cov_fixed
        List of fixed covariates. Default is `["intercept"]`.
    cov_exploring
        List of covariates to explore. Default is an empty list.

    """

    cov_fixed: list[str] = ["intercept"]
    cov_exploring: list[str] = []


class RoverFit(Config):
    """Rover fit function arguments.

    For more details please check out the ModRover package
    `documentation <https://ihmeuw-msca.github.io/modrover/>`_.

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
    """Rover covariate selection stage configuration.

    Parameters
    ----------
    groupby
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list.
    max_attempts
        Maximum number of attempts to run the Jobmon modeling tasks
        associated with the stage. Default is 1.
    t_threshold
        T-statistic threshold to consider as a covariate selection
        criterion. Default is 1.0.
    rover
        Rover class initialization arguments.
    rover_fit
        Rover fit function arguments.

    Notes
    -----
    If a StageConfig object is created while initializing an instance of
    OneModConfig, the onemod groupby setting will be added to the stage
    groupby setting.

    Examples
    --------
    All of the rover fields have default values equivalent to the
    following configuration.

    .. code-block:: yaml

        rover_covsel:
          groupby: []
          max_attempts: 1
          t_threshold: 1.0
          rover:
            cov_fixed: ["intercept"]
            cov_exploring: []
          rover_fit:
            strategies: ["forward"]
            top_pct_score: 0.1
            top_pct_learner: 1.0
            coef_bounds: {}

    """

    rover: RoverInit = RoverInit()
    rover_fit: RoverFit = RoverFit()
    t_threshold: float = 1.0
