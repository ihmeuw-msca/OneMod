from onemod.schema.base import Config, StageConfig


class DimensionInit(Config):
    """Weave dimension creation configuration. For more details please check
    `weave <https://ihmeuw-msca.github.io/weighted-average/>`_.

    Parameters
    ----------
    name
        Name of the dimension.
    coordinates
        Coordinates of the dimension. Default is None.
    kernel
        Kernel function to use. Default is "identity".
    distance
        Distance function to use. Default is None.
    radius
        List of radius for the kernel. Default is [0.0]. All the parameters in
        the list will be used to create different configurations of the
        dimension to do parameter ensemble.
    exponent
        List of exponent for the kernel. Default is [0.0]. All the parameters in
        the list will be used to create different configurations of the
        dimension to do parameter ensemble.
    version
        Version of the kernel. Default is None.
    distance_dict
        Dictionary of distance functions. Default is None.

    """

    name: str
    coordinates: str | list[str] | None = None
    kernel: str = "identity"
    distance: str | None = None
    radius: float | list[float] = [0.0]
    exponent: float | list[float] = [0.0]
    version: str | None = None
    distance_dict: str | list[str] | None = None


class WeaveConfig(StageConfig):
    """Configuration for the WeAve stage. In the actual OneMod configuration,
    this will be considered as a sub-model configuration for the weave stage.

    Parameters
    ----------
    max_batch
        Maximum batch size to use. Default is 5000. This overwrite the parent
        class default.
    dimensions
        Dictionary of dimension configurations. Default is an empty dictionary.

    Example
    -------
    All of the fields have default values. However using the default will not
    provide any information to the model run. Here is an example to setup the
    weave stage properly.

    .. code-block:: yaml

        weave:
          model1:
            groupby: [sex_id, super_region_id]
            max_attempts: 1
            max_batch: 5000
            dimensions:
              age:
                name: age_group_id
                coordinates: age_mid
                kernel: exponential
                radius: [0.75, 1, 1.25]
              year:
                name: year_id
                kernel: tricubic
                exponent: [0.5, 1, 1.5]
              location:
                name: location_id
                coordinates: [super_region_id, region_id, location_id]
                kernel: depth
                radius: [0.7, 0.8, 0.9]
            model2:
            groupby: [age_group_id, sex_id]
            max_attempts: 1
            max_batch: 5000
            dimensions:
              year:
                name: year_id
                kernel: tricubic
                exponent: [0.5, 1, 1.5]
              location:
                name: location_id
                kernel: identity
                distance: dictionary
                distance_dict: [/path/to/distance_dict1.parquet, /path/to/distance_dict2.parquet]

    """

    max_batch: int = 5000
    dimensions: dict[str, DimensionInit] = {}
