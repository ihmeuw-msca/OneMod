"""Configuration classes.

The configuration classes contain all of the settings needed to run a
stage (i.e., they don't include settings needed to set up a workflow).
Settings from PipelineConfig are passed to StageConfig when calling
Pipeline.add_stage().

"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Base configuration class."""

    model_config = ConfigDict(validate_assignment=True)

    def get(self, key: str, default: Any = None) -> Any:
        if not self.__contains__(key):
            return default
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> Any:
        if not self.__contains__(key):
            raise KeyError(f"invalid key: {key}")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.__setattr__(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields


class PipelineConfig(Config):
    """Pipeline configuration class."""

    ids: set[str]
    obs: str = "obs"
    pred: str = "pred"
    weights: str = "weights"
    test: str = "test"
    holdouts: set[str] = set()
    mtype: Literal["binomial", "gaussian", "poisson"]


class StageConfig(Config):
    """Stage configuration class."""

    model_config = ConfigDict(extra="allow")

    ids: set[str] | None = None
    obs: str | None = None
    pred: str | None = None
    weights: str | None = None
    test: str | None = None
    holdouts: set[str] | None = None
    mtype: Literal["binomial", "gaussian", "poisson"] | None = None

    def update(self, config: PipelineConfig) -> None:
        """Inherit settings from pipeline."""
        for key, value in config.model_dump().items():
            if self[key] is None:
                self[key] = value


class GroupedConfig(StageConfig):
    """Grouped stage configuration class."""

    data: Path | None = None


class CrossedConfig(StageConfig):
    """Crossed stage configuration class."""

    _crossable_params: set[str] = set()  # defined by class

    @property
    def crossable_params(self) -> set[str]:
        return self._crossable_params


class ModelConfig(GroupedConfig, CrossedConfig):
    """Model stage configuration class."""

    pass
