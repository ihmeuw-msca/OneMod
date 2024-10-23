"""SpXMod regression stage settings.

TODO: Update spline config and stage code for spxmod package updates
TODO: Update docstrings to clarify what spxmod defaults are (e.g., lam, priors)

"""

from typing import Any, Literal

from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from onemod.config import Config, ModelConfig


class SpxmodDimensionConfig(Config):
    """SpXMod dimension settings.

    Attributes
    ----------
    name : str
        Column name in input data corresponding to the dimension.
    dim_type : str
        Dimension type; either 'categorical' or 'numerical'.

    See Also
    --------
    `spxmod.dimension.Dimension <https://github.com/ihmeuw-msca/spxmod/blob/v0.2.1/src/spxmod/dimension.py#L7>`_.

    """

    model_config = ConfigDict(frozen=True)

    name: str
    dim_type: Literal["categorical", "numerical"]


class SpxmodSpaceConfig(Config):
    """SpXMod space settings.

    Attributes
    ----------
    name : str or None, optional
        Space name. If None, the name is set to the product of the
        dimension names. Default is None.
    dims : set of SpXModDimensionConfig
        Set of dimension settings.

    See Also
    --------
    `spxmod.space.Space <https://github.com/ihmeuw-msca/spxmod/blob/v0.2.1/src/spxmod/space.py#L13>`_.

    """

    name: str | None = None
    dims: set[SpxmodDimensionConfig]

    def model_post_init(self, *args, **kwargs) -> None:
        """Set space name."""
        if self.name is None:
            self.name = "*".join(dim.name for dim in self.dims)


class SpxmodVariableConfig(Config):
    """SpXMod variable settings.

    Attributes
    ----------
    name : str
        Column name in input data corresponding to the variable.
    space : str, SpxmodSpaceConfig, or None, optional
        Space to partition the variable on. If None, variable is not
        partitioned. Default is None.
    lam: nonnnegative float, dict, or None, optional
        Regularization parameter for the coefficients in the variable
        group. If the dimension is numerical, a Gaussian prior with mean
        0 and standard deviation 1/sqrt(lam) is set on the differences
        between neighboring coefficients along the dimension. If the
        dimension is categorical, a Gaussian prior with mean 0 and
        standard deviation 1/sqrt(lam) is set on the coefficients.
        Default is None.
    lam_mean : nonnegative float, optional
        Regularization parameter for the mean of the coefficients in the
        variable group. A Gaussian prior with mean 0 and standard
        deviation 1/sqrt(lam_mean) is set on the mean of the
        coefficients. Default is 0.
    gprior : dict of str: float or None, optional
        Gaussian prior for the variable. Argument is overwritten with
        (0, 1/sqrt(lam)) if dimension is categorical. Default is None.
    uprior : dict or None, optional
        Uniform prior for the variable. Default is None.
    scale_by_distance: bool, optional
        Whether to scale the prior standard deviation by the distance
        between the neighboring values along the dimension. For
        numerical dimensions only. Default is False.

    See Also
    --------
    `spxmod.variable_builder.VariableBuilder <https://github.com/ihmeuw-msca/spxmod/blob/v0.2.1/src/spxmod/variable_builder.py#L10>`_.

    """

    name: str
    space: str | SpxmodSpaceConfig | None = None
    lam: float | dict[str, float] | None = None
    lam_mean: float = Field(ge=0, default=0)
    gprior: dict[str, float] | None = None
    uprior: dict[str, float] | None = None
    scale_by_distance: bool = False


class SpxmodSplineConfig(Config):
    """SpXMod spline settings.

    Attributes
    ----------
    name : str
        Column name in input data corresponding to the spline variable.
    knots : list of float in [0, 1]
        Relative knot locations, including the boundaries. These are
        scaled to the range of the column values specified by `name`
        when building the spline basis.
    degree : nonnegative int, optional
        Spline degree. Default is 2.
    l_linear, r_linear : bool, optional
        Whether to use left or right linear tails. Default is False.
    include_first_basis: bool, optional
        Whether to use the first basis function. Default is True.

    See Also
    --------
    `xspline.XSpline <https://github.com/ihmeuw-msca/xspline/blob/v0.0.7/src/xspline/core.py#L272>`_.

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
    name in the `xmodel` configuration (the actual number of spline
    coefficients will depend on the `degree` and `knots` settings in
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

    """

    name: str
    knots: list[Annotated[float, Field(ge=0, le=1)]]
    degree: int = Field(gt=0, default=2)
    l_linear: bool = False
    r_linear: bool = False
    include_first_basis: bool = True


class SpxmodModelConfig(Config):
    """SPxMod model settings.

    Attributes
    ----------
    spaces : set[SpxmodSpaceConfig], optional
        Set of SpXMod space settings. Default is an empty set.
    variables : set[SpxmodVariableConfig]
        Set of SpXMod variable settings.
    param_specs : dict, optional
        Additional parameter specifications for the model.
        This argument is used `here https://github.com/ihmeuw-msca/regmod/blob/release/0.1.2/src/regmod/models/model.py#L133>`_.
        This is used for `inv_link` function, priors, or any other
        other settings that are captured by the current schema.
        Default is an empty dictionary.
    spline_config : SpXModSplineConfig or None, optional
        Spline variable settings. Currently, at most one spline variable
        is allowed. Default is None.
    lam : nonnegative float, optional
        Default lam value for all variables. Default is 0.

    See Also
    --------
    `spxmod.model.XModel <https://github.com/ihmeuw-msca/spxmod/blob/v0.2.1/src/spxmod/model.py#L74>`_

    """

    spaces: list[SpxmodSpaceConfig] = []
    variables: list[SpxmodVariableConfig]
    param_specs: dict[str, Any] = {}
    spline_config: SpxmodSplineConfig | None = None
    lam: float = Field(ge=0, default=0)


class SpxmodConfig(ModelConfig):
    """SpXMod regression stage settings.

    For more details, please check out the SpXMod package
    `documentation <https://github.com/ihmeuw-msca/spxmod/tree/v0.2.1>`_.

    Attributes
    ----------
    xmodel : SpxmodModelConfig
        SpXMod model settings.
    xmodel_fit : dict, optional
        Model fit function arguments. Default is an empty dictionary.

    """

    xmodel: SpxmodModelConfig
    xmodel_fit: dict[str, Any] = {}
