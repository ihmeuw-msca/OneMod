from typing import Literal

from onemod.schema.base import Config, StageConfig


class SPxModDimension(Config):
    name: str
    dim_type: Literal["categorical", "numerical"]


class SPxModSpace(Config):
    name: str | None = None
    dims: list[SPxModDimension] | None = None


class SPxModVarBuilder(Config):
    name: str
    space: str | SPxModSpace = SPxModSpace()
    lam: float | dict[str, float] = 0.0
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
    weights
        Name of the weight column in the data. Default is "weight".
    param_specs
        Additional parameter specifications for the model.

    """

    spaces: list[SPxModSpace] = []
    var_builders: list[SPxModVarBuilder] = []
    param_specs: dict | None = None

    coef_bounds: dict[str, tuple[float, float]] = {}
    lam: float = 0.0


class SPxModConfig(StageConfig):
    """SPxMod stage configuration class.

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
