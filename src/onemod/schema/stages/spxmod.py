from typing import Literal

from onemod.schema.base import Config, StageConfig


class SPxModDimension(Config):
    """For details of the settings description please check
    `here <https://github.com/ihmeuw-msca/spxmod/blob/main/src/spxmod/dimension.py#L7>`_.

    """

    name: str
    dim_type: Literal["categorical", "numerical"]


class SPxModSpace(Config):
    """For details of the settings description please check
    `here <https://github.com/ihmeuw-msca/spxmod/blob/main/src/spxmod/space.py#L13>`_.

    """

    name: str | None = None
    dims: list[SPxModDimension] | None = None


class SPxModVarBuilder(Config):
    """For details of the settings description please check
    `here <https://github.com/ihmeuw-msca/spxmod/blob/main/src/spxmod/variable_builder.py#L10>`_.

    """

    name: str
    space: str | SPxModSpace = SPxModSpace()
    lam: float | dict[str, float] | None = None
    lam_mean: float = 0.0
    gprior: dict[str, float] | None = None
    uprior: dict[str, float] | None = None
    scale_by_distance: bool = False


class XModelInit(Config):
    """SPxMod class initialization arguments.

    To create a spxmod model, additional configuration args `mtype`,
    `obs`, `weights`, are taken from the OneMod config. For more details
    please check out the RegModSM package
    `documentation <https://github.com/ihmeuw-msca/spxmod>`_.

    Parameters
    ----------
    spaces
        List of dictionaries containing space names and arguments.
    var_builders
        List of dictionaries containing variable group names and arguments.
    param_specs
        Additional parameter specifications for the model.
        This argument is used `here <https://github.com/ihmeuw-msca/regmod/blob/release/0.1.2/src/regmod/models/model.py#L133>`_.
        This is used for inv_link function or linear prior or any other settings
        that are captured by the current schema.
    coef_bounds
        Dictionary containing bounds for the coefficients.
    lam
        Default lam value for all var_builders. Default is 0.0.

    """

    spaces: list[SPxModSpace] = []
    var_builders: list[SPxModVarBuilder] = []
    param_specs: dict | None = None
    coef_bounds: dict[str, dict[str, float]] = {}
    lam: float = 0.0


class SPxModConfig(StageConfig):
    """SPxMod stage configuration class.

    Additional configuration arg `mtype` taken from the OneMod config.

    Parameters
    ----------
    groupby
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list.
    max_attempts
        Maximum number of attempts to run the Jobmon modeling tasks
        associated with the stage. Default is 1.
    xmodel
        Model initialization arguments.
    xmodel_fit
        Model fit function arguments.

    Notes
    -----
    If a StageConfig object is created while initializing an instance of
    OneModConfig, the onemod groupby setting will be added to the stage
    groupby setting.

    Examples
    --------
    All of the spxmod fields have default values equivalent to the
    following configuration.

    .. code-block:: yaml

        spxmod:
          groupby: []
          max_attempts: 1
          xmodel:
            spaces: []
            var_builders: []
          xmodel_fit: {}

    """

    xmodel: XModelInit = XModelInit()
    xmodel_fit: dict = {}
