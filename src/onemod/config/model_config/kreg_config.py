"""KReg kernel regression stage settings.

TODO: Add descriptions and default values
TODO: Generalize KReg config
TODO: Add custom kernels

"""

from onemod.config import Config, ModelConfig


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


class KregConfig(ModelConfig):
    """KReg kernel regression stage settings.

    For more details, please check out the KReg package
    `documentation <https://github.com/ihmeuw-msca/kreg>`_.

    Attributes
    ----------
    kreg_model : KregModelConfig
        Description.
    kreg_fit : KregFitConfig
        Description.
    kreg_uncertainty : KregUncertaintyConfig, optional
        Description.

    """

    kreg_model: KregModelConfig
    kreg_fit: KregFitConfig
    kreg_uncertainty: KregUncertaintyConfig = KregUncertaintyConfig()
