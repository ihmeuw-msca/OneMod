from typing import Literal

from pydantic import model_validator

from onemod.schema.base import Config, StageConfig


class WeaveDimension(Config):
    """WeAve dimension configuration.

    Parameters
    ----------
    name
        Dimension name.
    coordinates
        Dimension coordinates. Default is None.
    kernel
        Kernel function name. Default is "identity".
    distance
        Distance function name. Default is None.
    radius
        List of radii for the exponential, depth, and inverse kernels.
        Default is None. These parameters are used to create different
        weave models for the ensemble.
    exponent
        List of tricubic kernel exponents. Default is None. These
        parameters are used to create different weave models for the
        ensemble.
    version
        Depth kernel version. Default is None.
    distance_dict
        List of filepaths (str) containing distance dictionaries.
        Default is None. These parameters are used to create different
        weave models for the ensemble.

    """

    name: str
    coordinates: list[str] | None = None
    kernel: Literal[
        "exponential", "tricubic", "depth", "inverse", "identity"
    ] = "identity"
    distance: Literal["euclidean", "tree", "dictionary"] | None = None
    radius: list[float] | None = None
    exponent: list[float] | None = None
    version: Literal["codem", "stgpr"] | None = None
    distance_dict: list[str] | None = None

    @model_validator(mode="after")
    def check_args(self):
        """Make sure each dimension has required parameters."""
        if (
            self.kernel in ["exponential", "depth", "inverse"]
            and self.radius is None
        ):
            raise ValueError(f"{self.kernel} kernel requires radius parameter")
        if self.kernel == "tricubic" and self.exponent is None:
            raise ValueError("tricubic kernel requires exponent parameter")
        if self.distance == "dictionary" and self.distance_dict is None:
            raise ValueError(
                "dictionary distance requires distance_dict parameter"
            )
        return self


class WeaveModel(StageConfig):
    """WeAve model configuration.

    Parameters
    ----------
    groupby
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list, which means all points are run in a single model.
    max_attempts
        Maximum number of attempts to run the Jobmon task associated
        with the stage. Default is 1.
    dimensions
        Dictionary of WeAve dimension configuration objects.
    max_batch
        Maximum number of points per batch when fitting the model using
        multiple Jobmon tasks. Default is -1, which means do not fit the
        model in batches.

    """

    dimensions: dict[str, WeaveDimension]
    max_batch: int = -1


class WeaveConfig(Config):
    """WeAve stage configuration. Unlike other OneMod stages, the WeAve
    stage can consist of multiple models which different parameters.

    Parameters
    ----------
    models
        Dictionary of WeAve model configuration objects.

    Examples
    --------
    Here is an example to set up the WeAve stage properly. For more
    details please check out the WeAve package
    `documentation <https://ihmeuw-msca.github.io/weighted-average/>`_.

    .. code-block:: yaml

        weave:
          models:
            super_region_model:
              groupby: [sex_id, super_region_id]
              max_attempts: 1
              max_batch: 5000
              dimensions:
                age:
                  name: age_group_id
                  coordinates: [age_mid]
                  kernel: exponential
                  radius: [0.75, 1, 1.25]
                location:
                  name: location_id
                  coordinates: [super_region_id, region_id, location_id]
                  kernel: depth
                  radius: [0.7, 0.8, 0.9]
                year:
                  name: year_id
                  kernel: tricubic
                  exponent: [0.5, 1, 1.5]
            age_model:
              groupby: [sex_id, age_group_id]
              max_attempts: 1
              dimensions:
                location:
                  name: location_id
                  kernel: identity
                  distance: dictionary
                  distance_dict:
                  - /path/to/distance_dict1.parquet
                  - /path/to/distance_dict2.parquet
                year:
                  name: year_id
                  kernel: inverse
                  radius: [0.5, 1, 1.5]

    """

    models: dict[str, WeaveModel]
