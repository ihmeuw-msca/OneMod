from typing import Any

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """An extension of BaseModel that supports some dict-like dunder methods.
    Froze model fields once set.

    """

    model_config = ConfigDict(frozen=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a field value by key. If key is not found, return default.

        Parameters
        ----------
        key
            Field name.
        default
            Default value to return if key is not found. Default is None.

        Returns
        -------
        Any
            Field value if key is found, otherwise default.

        """
        if not self.__contains__(key):
            return default
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> Any:
        if not self.__contains__(key):
            raise KeyError(f"Key {repr(key)} not found")
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields


class StageConfig(Config):
    """Stage configuration class. All stages configuration will include
    `groupby`, `max_attempts` and `max_batch` fields. All the stages will use
    the provided defaults unless they are overwritten.

    Parameters
    ----------
    groupby
        List of index columns to group by. Default is an empty list, which means
        all data points are run in a single model.
    max_attempts
        Maximum number of attempts to run the stage. Default is 1.
    max_batch
        Maximum number of data points to run in a single batch. If -1, run all
        models in a single batch.

    """

    groupby: list[str] = []
    max_attempts: int = 1
    max_batch: int = -1
