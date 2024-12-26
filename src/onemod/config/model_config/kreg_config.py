"""KReg kernel regression stage settings.

TODO: Remove any unused config items
TODO: Add descriptions and default values
TODO: Generalize KReg config
TODO: Add custom kernels

"""

from typing import Literal

from onemod.config import Config, StageConfig


class KregModelConfig(Config):
    """KReg model settings.

    Attributes
    ----------
    age_scale : float
        Description.
    gamma_age : float
        Description.
    gamma_year : float
        Description.
    exp_location : float
        Description.
    lam : float
        Description.
    nugget : float
        Description.

    """

    age_scale: float
    gamma_age: float
    gamma_year: float
    exp_location: float
    lam: float
    nugget: float


class KregFitConfig(Config):
    """KReg optimization settings.

    Attributes
    ----------
    gtol : float
        Description.
    max_iter : int
        Description.
    cg_maxiter : int
        Description.
    cg_maxiter_increment: int
        Description.
    nystroem_rank : int
        Description.

    """

    gtol: float
    max_iter: int
    cg_maxiter: int
    cg_maxiter_increment: int
    nystroem_rank: int


class KregUncertaintyConfig(Config):
    """KReg uncertainty settings.

    Attributes
    ----------
    num_samples : int
        Number of times to sample the model Hessian. Default is 50.
    save_draws : bool, optional
        Whether to save uncertainty draws. Default is False.
    lanczos_order : int, optional
        Maximum number of Lanczos steps. Default is 150.

    """

    num_samples: int = 50
    save_draws: bool = False
    lanczos_order: int = 150


class KregConfig(StageConfig):
    """KReg kernel regression stage settings.

    For more details, please check out the KReg package
    `documentation <https://github.com/ihmeuw-msca/kreg>`_.

    Attributes
    ----------
    id_columns : set[str]
        ID column names, e.g., 'age_group_id', 'location_id', 'sex_id',
        or 'year_id'. ID columns should contain nonnegative integers.
    model_type : str
        Model type; either 'binomial', 'gaussian', or 'poisson'.
    observation_column : str, optional
        Observation column name for pipeline input. Default is 'obs'.
    prediction_column : str, optional
        Prediction column name for pipeline output. Default is 'pred'.
    weights_column : str, optional
        Weights column name for pipeline input. The weights column
        should contain nonnegative floats. Default is 'weights'.
    test_column : str, optional
        Test column name. The test column should contain values 0
        (train) or 1 (test). The test set is never used to train stage
        models, so it can be used to evaluate out-of-sample performance
        for the entire pipeline. If no test column is provided, all
        missing observations will be treated as the test set. Default is
        'test'.
    holdout_columns : set[str] or None, optional
        Holdout column names. The holdout columns should contain values
        0 (train), 1 (holdout), or NaN (missing observations). Holdout
        sets are used to evaluate stage model out-of-sample performance.
        Default is None.
    coef_bounds : dict or None, optional
        Dictionary of coefficient bounds with entries
        cov_name: (lower, upper). Default is None.
    kreg_model : KregModelConfig
        Description.
    kreg_fit : KregFitConfig
        Description.
    kreg_uncertainty : KregUncertaintyConfig, optional
        Description.

    """

    id_columns: set[str]
    model_type: Literal["binomial", "gaussian", "poisson"]
    observation_column: str = "obs"
    prediction_column: str = "pred"
    weights_column: str = "weights"
    test_column: str = "test"
    holdout_columns: set[str] | None = None
    coef_bounds: dict[str, tuple[float, float]] | None = None
    kreg_model: KregModelConfig
    kreg_fit: KregFitConfig
    kreg_uncertainty: KregUncertaintyConfig = KregUncertaintyConfig()
