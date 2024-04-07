from typing import Any

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """An extension of BaseModel that supports some dict-like dunder methods.
    Froze model fields once set.

    """

    model_config = ConfigDict(frozen=True)

    def get(self, key: str, default: Any = None) -> Any:
        if self.__contains__(key):
            return default
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields


class StageConfig(Config):
    """Stage configuration class. All stages configuration will include `groupby`,
    `max_attempts` and `max_batch` fields. And they all have their defaults.

    """

    groupby: list[str] = []
    max_attempts: int = 1
    max_batch: int = -1
