"""Configuration classes."""

from abc import ABC
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Config(BaseModel, ABC):
    """Base configuration class."""

    model_config = ConfigDict(validate_assignment=True, protected_namespaces=())

    def get(self, key: str, default: Any = None) -> Any:
        if not self.__contains__(key):
            return default
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> Any:
        if not self.__contains__(key):
            raise KeyError(f"invalid config item: {key}")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.__setattr__(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields


class PipelineConfig(Config):
    """Pipeline configuration class.

    Attributes
    ----------
    id_columns : set[str]
        ID column names, e.g., 'age_group_id', 'location_id', 'sex_id',
        or 'year_id'. ID columns should contain nonnegative integers.
    model_type : str
        Model type; either 'binomial', 'gaussian', or 'poisson'.
    observation_column : str, optional
        Observation column name for pipeline input. Default is 'obs'.
    prediction_column : str, optional
        Prediction column name for pipeline output. Default is 'pred'.
    weights_column : str, optional
        Weights column name for pipeline input. The weights column
        should contain nonnegative floats. Default is 'weights'.
    test_column : str, optional
        Test column name. The test column should contain values 0
        (train) or 1 (test). The test set is never used to train stage
        models, so it can be used to evaluate out-of-sample performance
        for the entire pipeline. If no test column is provided, all
        missing observations will be treated as the test set. Default is
        'test'.
    holdout_columns : set[str], optional
        Holdout column names. The holdout columns should contain values
        0 (train), 1 (holdout), or NaN (missing observations). Holdout
        sets are used to evaluate stage model out-of-sample performance.
        Default is an empty set.
    coef_bounds : dict, optional
        Dictionary of coefficient bounds with entries
        cov_name: (lower, upper). Default is an empty dictionary.

    """

    id_columns: set[str]
    model_type: Literal["binomial", "gaussian", "poisson"]
    observation_column: str = "obs"
    prediction_column: str = "pred"
    weight_column: str = "weights"
    test_column: str = "test"
    holdout_columns: set[str] = set()
    coef_bounds: dict[str, tuple[float, float]] = {}


class StageConfig(Config):
    """Stage configuration class.

    Settings from PipelineConfig are passed to StageConfig when calling
    Pipeline.add_stage().

    Attributes
    ----------
    id_columns : set[str] or None, optional
        ID column names, e.g., 'age_group_id', 'location_id', 'sex_id',
        or 'year_id'. ID columns should contain nonnegative integers.
        Default is None.
    model_type : str or None, optional
        Model type; either 'binomial', 'gaussian', or 'poisson'.
        Default is None.
    observation_column : str or None, optional
        Observation column name for pipeline input. Default is None.
    prediction_column : str or None, optional
        Prediction column name for pipeline output. Default is None.
    weights_column : str or None, optional
        Weights column name for pipeline input. The weights column
        should contain nonnegative floats. Default is None.
    test_column : str or None, optional
        Test column name. The test column should contain values 0
        (train) or 1 (test). The test set is never used to train stage
        models, so it can be used to evaluate out-of-sample performance
        for the entire pipeline. If no test column is provided, all
        missing observations will be treated as the test set. Default is
        None.
    holdout_columns : set[str] or None, optional
        Holdout column names. The holdout columns should contain values
        0 (train), 1 (holdout), or NaN (missing observations). Holdout
        sets are used to evaluate stage model out-of-sample performance.
        Default is None.
    coef_bounds : dict or None, optional
        Dictionary of coefficient bounds with entries
        cov_name: (lower, upper). Default is None.

    """

    model_config = ConfigDict(extra="allow")

    id_columns: set[str] | None = None
    model_type: Literal["binomial", "gaussian", "poisson"] | None = None
    observation_column: str | None = None
    prediction_column: str | None = None
    weight_column: str | None = None
    test_column: str | None = None
    holdout_columns: set[str] | None = None
    coef_bounds: dict[str, tuple[float, float]] | None = None

    def update(self, config: PipelineConfig) -> None:
        """Inherit settings from pipeline."""
        for key, value in config.model_dump().items():
            if self[key] is None:
                self[key] = value


class ModelConfig(StageConfig):
    """Model stage configuration class.

    Attributes
    ----------
    data : Path or None, optional
        Path to input data. Required for `groupby`. Default is None.

    """

    data: Path | None = None
    _crossable_params: set[str] = set()  # defined by class

    @property
    def crossable_params(self) -> set[str]:
        return self._crossable_params
