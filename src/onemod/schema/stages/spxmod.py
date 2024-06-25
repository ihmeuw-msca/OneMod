from typing import Literal

from pydantic import Field
from typing_extensions import Annotated

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


class SPxModSplineConfig(Config):
    """Spline variable configuration.

    Parameters
    ----------
    name : str
        Column name in input data corresponding to the spline variable.
    knots : list of float in [0, 1]
        Relative knot locations, including the boundaries. These are
        scaled to the range of the column values specified by 'name'
        when building the spline basis.
    degree : non-negative int, optional
        Degree of the spline. Default is 2.
    l_linear, r_linear : bool, optional
        Whether to use left or right linear tails. Default is False.
    include_first_basis : bool, optional
        Description. Default is True.

    See Also
    --------
    `XSpline https://github.com/ihmeuw-msca/xspline/blob/2302f11419e8b13305c93850ecd6df9045390dd6/src/xspline/core.py#L276>`_.

    Examples
    --------
    Below is an example configuration for a quadratic spline on
    'year_id' with five equally spaced knots. Because `r_linear` is
    true, the spline is linear between the last two knots (i.e., a right
    linear tail).

    .. code-block:: yaml

        spline_config:
            name: year_id
            knots: [0.0, 0.25, 0.5, 0.75, 1.0]
            degree: 2
            l_linear: false
            r_linear: true
            include_first_basis: false

    To specify a spline variable, use 'spline' as the `var_builder`
    name in the `xmodel` configuartion (the actual number of spline
    coefficients will depend upon the `degree` and `knots` settings in
    `spline_config`). In the example below, each location has its own
    intercept and spline variable on 'year_id', with Gaussian priors
    with mean 0 and standard deviation 1/sqrt(lam) set on the mean of
    the variable coefficients so that the level and shape of the curve
    vary smoothly by location. If you include both an intercept and
    spline variable for the same space, as in this example, you should
    set `include_first_basis` to false in `spline_config`.

    .. code-block:: yaml

        xmodel:
          spaces:
          - name: location_id
            dims:
            - name: location_id
              dim_type: categorical
          var_builders:
          - name: intercept
            space: location_id
            lam: 1.0
          - name: spline
            space: location_id
            lam: 10.0
          ...

    """

    name: str
    knots: list[Annotated[float, Field(ge=0, le=1)]]
    degree: int = Field(gt=0, default=2)
    l_linear: bool = False
    r_linear: bool = False
    include_first_basis: bool = True


class XModelInit(Config):
    """SPxMod class initialization arguments.

    To create a spxmod model, additional configuration args `mtype`,
    `obs`, `weights`, are taken from the OneMod config. For more details
    please check out the SPxMod package
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
        This is used for inv_link function or linear prior or any other
        settings that are captured by the current schema.
    coef_bounds
        Dictionary containing bounds for the coefficients.
    spline_config
        Dictionary containing spline variable configuration.
    lam
        Default lam value for all var_builders. Default is 0.0.

    # TODO: Should rover and spxmod coef_bounds be moved to onemod config?

    """

    spaces: list[SPxModSpace] = []
    var_builders: list[SPxModVarBuilder] = []
    param_specs: dict | None = None
    coef_bounds: dict[str, dict[str, float]] = {}
    spline_config: SPxModSplineConfig | None = None
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
    * If a StageConfig object is created while initializing an instance
      of OneModConfig, the onemod groupby setting will be added to the
      stage groupby setting.
    * `xmodel_fit` options such as tolerance values should not be
      written in scientific notation, as pydantic will read it as a
      string (e.g., use 0.0001 instead of 1e-4).

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
            coef_bounds: {}
            lam: 0.0
          xmodel_fit: {}

    """

    xmodel: XModelInit = XModelInit()
    xmodel_fit: dict = {}
