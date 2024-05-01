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
    space: SPxModSpace = SPxModSpace()
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

    spaces: list[SPxModSpace] = []
    var_builders: list[SPxModVarBuilder] = []
    param_specs: dict | None = None


class SPxModConfig(StageConfig):
    """SPxMod stage configuration class.

    Additional configuration arg `mtype` taken from the OneMod config.

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
