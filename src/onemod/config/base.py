"""Configuration classes."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Base configuration class.

    Config instances are dictionary-like objects that contains settings.
    For attribute validation, users can create custom configuration
    classes by subclassing Config. Alternatively, users can add extra
    attributes to Config instances without validation.

    """

    model_config = ConfigDict(
        extra="allow", validate_assignment=True, protected_namespaces=()
    )

    def get(self, key: str, default: Any = None) -> Any:
        if self.__contains__(key):
            return getattr(self, key)
        return default

    def __getitem__(self, key: str) -> Any:
        if self.__contains__(key):
            return getattr(self, key)
        raise KeyError(f"Invalid config item: {key}")

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields or key in self.model_extra


class StageConfig(Config):
    """Stage configuration class.

    StageConfig instances are dictionary-like objects that contains
    settings that apply to a particular pipeline stage. For attribute
    validation, users can create custom stage configuration classes by
    subclassing StageConfig. Alternatively, users can add extra
    attributes to StageConfig instances without validation.

    If a StageConfig instances does not contain an attribute, the get
    and __getitem__ methods will return the corresponding pipeline
    attribute, if it exists.

    """

    model_config = ConfigDict(
        extra="allow", validate_assignment=True, protected_namespaces=()
    )

    _pipeline_config: Config = Config()
    _crossable_params: set[str] = set()  # TODO: unique list

    @property
    def pipeline_config(self) -> Config:
        return self._pipeline_config

    @pipeline_config.setter
    def pipeline_config(self, config: Config | dict) -> None:
        if isinstance(config, dict):
            config = Config(**config)
        self._pipeline_config = config

    @property
    def crossable_params(self) -> set[str]:
        return self._crossable_params

    def stage_contains(self, key: str) -> bool:
        return key in self.model_fields or key in self.model_extra

    def pipeline_contains(self, key: str) -> bool:
        return key in self.pipeline_config

    def get_from_stage(self, key: str, default: Any = None) -> Any:
        if self.stage_contains(key):
            return getattr(self, key)
        return default

    def get_from_pipeline(self, key: str, default: Any = None) -> Any:
        return self.pipeline_config.get(key, default)

    def get(self, key: str, default: Any = None) -> Any:
        if self.stage_contains(key):
            return getattr(self, key)
        return self.pipeline_config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if self.stage_contains(key):
            return getattr(self, key)
        return self.pipeline_config[key]

    def __contains__(self, key: str) -> bool:
        return self.stage_contains(key) or self.pipeline_contains(key)
