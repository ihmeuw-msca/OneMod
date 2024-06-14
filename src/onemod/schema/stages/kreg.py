"""Kernel regression settings.

TODO: Generalize parameters
TODO: Create class for kernels?
TODO: Ask Alex for descriptions, types, defaults

"""

from onemod.schema.base import Config, StageConfig


class KregModel(Config):
    """Kernel regression model arguments.

    Parameters
    ----------
    age_scale
        Description.
    gamma_age
        Description.
    gamma_year
        Description.
    exp_location
        Description.
    lam
        Description.
    nugget
        Description.

    """

    age_scale: float
    gamma_age: float
    gamma_year: float
    exp_location: float
    lam: float
    nugget: float


class KregFit(Config):
    """Kernel regression optimization arguments.

    Parameters
    ----------
    gtol
        Description.
    max_iter
        Description.
    cg_maxiter
        Description.
    cg_maxiter_increment
        Description.
    nystroem_rank
        Description.

    """

    gtol: float
    max_iter: int
    cg_maxiter: int
    cg_maxiter_increment: int
    nystroem_rank: int


class KregUncertainty(Config):
    """Kernel regression uncertainty arguments.

    Parameters
    ----------
    num_samples
        Description.
    lanczos_order
        Description.

    """

    num_samples: int = 50
    lanczos_order: int = 150


class KregConfig(StageConfig):
    """Kernel regression stage configuration.

    Parameters
    ----------
    groupby
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list.
    max_attempts
        Maximum number of attempts to run the Jobmon modeling tasks
        associated with the stage. Default is 1.
    kreg_model
        Description.
    kreg_fit
        Description.

    Notes
    -----
    If a StageConfig object is created while initializing an instance of
    OneModConfig, the onemod groupby setting will be added to the stage
    groupby setting.

    Examples
    --------

    """

    kreg_model: KregModel
    kreg_fit: KregFit
    kreg_uncertainty: KregUncertainty = KregUncertainty()
