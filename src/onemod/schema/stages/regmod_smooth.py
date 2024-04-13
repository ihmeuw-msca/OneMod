from onemod.schema.base import Config, StageConfig


class ModelInit(Config):
    """Arguments for initialiation of a regmodsm model. To create a regmodsm
    model, we need additional configuration `mtype`, `obs`, `weights`, from the
    global config. For more details please check
    `regmodsm <https://github.com/ihmeuw-msca/regmodsm>`_.

    Parameters
    ----------
    dims
        List of dictionaries with dimension information. Default is an empty
        list. When `age_mid` as a dimension is not provided. It will be
        automatically created and added to the list.
    var_groups
        List of dictionaries with variable group information. Default is an
        empty list. This list is considered as a suppliment list to the selected
        covariates with dimension `age_mid` from `rover_covsel` stage. Those
        variables will be automatically created and added to the list.
    coef_bounds
        Dictionary of coefficient bounds. Default is an empty dictionary.
    lam
        Default smoothing parameter, you can overwrite this value when define
        variable groups in `var_groups`. Default is 0.0.

    """

    dims: list[dict] = []
    var_groups: list[dict] = []

    coef_bounds: dict[str, tuple[float, float]] = {}
    lam: float = 0.0


class RegmodSmoothConfig(StageConfig):
    """RegmodSmooth stage configuration class. Will need the `mtype` form the
    upper level to initialize a regmodsm model.

    Parameters
    ----------
    xmodel
        Model initialization arguments.
    xmodel_fit
        Model fit function arguments.

    Example
    -------
    All of the fields have default values. And it is equivalent to the following
    configuration for the regmod_smooth section.

    .. code-block:: yaml

        regmod_smooth:
          groupby: []
          max_attempts: 1
          max_batch: -1
          xmodel:
            dims: []
            var_groups: []
            coef_bounds: {}
            lam: 0.0
          xmodel_fit: {}

    """

    xmodel: ModelInit = ModelInit()
    xmodel_fit: dict = {}
