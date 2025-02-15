"""Configuration classes."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from onemod.dtypes.unique_sequence import unique_list


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
        return key in self._get_fields()

    def __repr__(self) -> str:
        arg_list = []
        for key in self._get_fields():
            arg_list.append(f"{key}={getattr(self, key)!r}")
        return f"{type(self).__name__}({', '.join(arg_list)})"

    def _get_fields(self) -> list[str]:
        return list(self.model_dump(exclude_none=True))


class StageConfig(Config):
    """Stage configuration class.

    If a StageConfig instance does not contain an attribute, the get and
    __getitem__ methods will return the corresponding pipeline
    attribute, if it exists.

    """

    model_config = ConfigDict(
        extra="allow", validate_assignment=True, protected_namespaces=()
    )

    _pipeline_config: Config = Config()
    _required: list[str] = []

    def add_pipeline_config(self, pipeline_config: Config | dict) -> None:
        if isinstance(pipeline_config, dict):
            pipeline_config = Config(**pipeline_config)

        missing = []
        for item in self._required:
            if not self.stage_contains(item) and item not in pipeline_config:
                missing.append(item)
        if missing:
            raise AttributeError(f"Missing required config items: {missing}")

        self._pipeline_config = pipeline_config

    def get(self, key: str, default: Any = None) -> Any:
        if self.stage_contains(key):
            return getattr(self, key)
        return self._pipeline_config.get(key, default)

    def get_from_stage(self, key: str, default: Any = None) -> Any:
        if self.stage_contains(key):
            return getattr(self, key)
        return default

    def get_from_pipeline(self, key: str, default: Any = None) -> Any:
        return self._pipeline_config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if self.stage_contains(key):
            return getattr(self, key)
        return self._pipeline_config[key]

    def __contains__(self, key: str) -> bool:
        return self.stage_contains(key) or self.pipeline_contains(key)

    def stage_contains(self, key: str) -> bool:
        return key in self._get_stage_fields()

    def pipeline_contains(self, key: str) -> bool:
        return key in self._pipeline_config

    def _get_fields(self) -> list[str]:
        return unique_list(
            self._get_stage_fields() + self._get_pipeline_fields()
        )

    def _get_stage_fields(self) -> list[str]:
        return list(self.model_dump(exclude_none=True))

    def _get_pipeline_fields(self) -> list[str]:
        return self._pipeline_config._get_fields()

    def __repr__(self) -> str:
        arg_list = []
        for key in self._get_fields():
            arg_list.append(f"{key}={self.get(key)!r}")
        return f"{type(self).__name__}({', '.join(arg_list)})"
