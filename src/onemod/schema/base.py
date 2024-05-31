from typing import Any

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """An extension of BaseModel that supports some dict-like dunder
    methods. Model fields frozen once set.

    """

    model_config = ConfigDict(frozen=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a field value by key. If key not found, return default.

        Parameters
        ----------
        key
            Field name.
        default
            Default value to return if key not found. Default is None.

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
    """Stage configuration class.

    All stage configurations will include `groupby` and `max_attempts`
    fields. All stages will use the provided defaults unless they are
    overwritten.

    Parameters
    ----------
    groupby
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list.
    max_attempts
        Maximum number of attempts to run the Jobmon modeling tasks
        associated with the stage. Default is 1.

    Notes
    -----
    If a StageConfig object is created while initializing an instance of
    OneModConfig, the onemod groupby setting will be added to the stage
    groupby setting.

    """

    groupby: list[str] = []
    max_attempts: int = 1
