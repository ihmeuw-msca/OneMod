from onemod.schema.base import StageConfig


class KregConfig(StageConfig):
    """Description.

    TODO: Generalize parameters.
    TODO: Ask Alex for descriptions, types, defaults

    Parameters
    ----------
    groupby
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list.
    max_attempts
        Maximum number of attempts to run the Jobmon modeling tasks
        associated with the stage. Default is 1.
    gamma_age
        Description.
    gamma_year
        Description.
    alpha_year
        Description.
    exp_location
        Description.
    exp_sex
        Description.
    lam
        Description.
    nugget
        Description.

    """

    gamma_age: float
    gamma_year: float
    alpha_year: float
    exp_location: float
    exp_sex: float
    lam: float
    nugget: float
