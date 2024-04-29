from onemod.schema.base import Config, StageConfig


class ModelInit(Config):
    """RegmodSmooth class initialization arguments.

    To create a regmodsm model, additional configuration args `mtype`,
    `obs`, `weights`, are taken from the OneMod config. For more details
    please check out the RegModSM package
    `documentation <https://github.com/ihmeuw-msca/regmodsm>`_.

    Parameters
    ----------
    dims
        List of dictionaries with dimension information. Default is an
        empty list. When `age_mid` as a dimension is not provided. It
        will be automatically created and added to the list.
    var_groups
        List of dictionaries with variable group information. Default is
        an empty list. This list is considered as a suppliment list to
        the selected covariates with dimension `age_mid` from
        `rover_covsel` stage. Those variables will be automatically
        created and added to the list.
    coef_bounds
        Dictionary of coefficient bounds. Default is an empty
        dictionary.
    lam
        Default smoothing parameter, you can overwrite this value when
        define variable groups in `var_groups`. Default is 0.0.

    """

    dims: list[dict] = []
    var_groups: list[dict] = []

    coef_bounds: dict[str, tuple[float, float]] = {}
    lam: float = 0.0


class RegmodSmoothConfig(StageConfig):
    """RegmodSmooth stage configuration class.

    Additional configuration arg `mtype` taken from the OneMod config.

    Parameters
    ----------
    groupby
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list, which means all points are run in a single model.
    max_attempts
        Maximum number of attempts to run the Jobmon task associated
        with the stage. Default is 1.
    xmodel
        Model initialization arguments.
    xmodel_fit
        Model fit function arguments.

    Example
    -------
    All of the regmodsm fields have default values equivalent to the
    following configuratio.

    .. code-block:: yaml

        regmod_smooth:
          groupby: []
          max_attempts: 1
          xmodel:
            dims: []
            var_groups: []
            coef_bounds: {}
            lam: 0.0
          xmodel_fit: {}

    """

    xmodel: ModelInit = ModelInit()
    xmodel_fit: dict = {}
