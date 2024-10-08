"""Input/output classes.

Notes
-----
* Classes to organize stage input and output
* Data class is just a placeholder for new onemod data types
* Input and output treated like a dictionary
  * Provides validation
  * We can create subclasses for onemod stages
  * User can create their own custom subclasses, but they don't need to
* How can we define serialization to not show data field?
* Lots of shared methods with Config...

"""

from abc import ABC
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from onemod.types import Data


class IO(BaseModel, ABC):
    """Stage input/output base class."""

    model_config = ConfigDict(validate_assignment=True)

    stage: str
    data: dict[str, Path | Data] = Field(default={}, exclude=True)

    @staticmethod
    def deserialize_item(value: dict) -> Any:
        """Helper function to deserialize an individual input/output."""
        if isinstance(value, dict) and value.get("type") == "Data":
            return Data.from_dict(value["data"])
        return value

    def get(self, key: str, default: Any = None) -> Any:
        if not self.__contains__(key):
            return default
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> Data:
        if not self.__contains__(key):
            raise KeyError(f"invalid key: {key}")
        return self.data[key]

    def __setitem__(self, key: str, value: Data) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data


class Input(IO):
    """Stage input class."""

    required: dict[str, Data] = {}  # defined by class
    optional: dict[str, Data] = {}  # defined by class

    @computed_field
    @property
    def dependencies(self) -> set[str]:
        return set(
            value.stage
            for value in self.data.values()
            if isinstance(value, Data)
        )
        
    @classmethod
    def from_dict(cls, data_dict: dict) -> 'Input':
        """Reconstruct an Input object from a dictionary."""
        required = {
            key: cls.deserialize_item(value)
            for key, value in data_dict.get("required", {}).items()
        }
        optional = {
            key: cls.deserialize_item(value)
            for key, value in data_dict.get("optional", {}).items()
        }
        return cls(required=required, optional=optional)

    def to_dict(self) -> dict:
        """Convert Input to a dictionary."""
        return {
            "required": {
                key: value.to_dict() if isinstance(value, Data) else value
                for key, value in self.required.items()
            },
            "optional": {
                key: value.to_dict() if isinstance(value, Data) else value
                for key, value in self.optional.items()
            }
        }

    def validate(self) -> None:
        self._check_missing()
        self._check_types()

    def _check_missing(self) -> None:
        missing_input = [key for key in self.required if key not in self._data]
        if missing_input:
            raise KeyError(
                f"{self.stage} missing required input: {missing_input}"
            )

    def _check_types(self) -> None:
        wrong_types = []
        for key, value in self._required.items():
            if not isinstance(self._data[key], value):
                wrong_types.append(key)
        for key, value in self._optional.items():
            if not isinstance(self._data[key], value):
                wrong_types.append(key)
        if wrong_types:
            raise TypeError(
                f"{self.stage} input has incorrect types: {wrong_types}"
            )


class Output(IO):
    """Stage output class."""

    data: dict[str, Data] = Field(default={}, exclude=True)

    @classmethod
    def from_dict(cls, data_dict: dict) -> 'Output':
        """Reconstruct an Output object from a dictionary."""
        data = {
            key: cls.deserialize_item(value)
            for key, value in data_dict.get("data", {}).items()
        }
        return cls(data=data)

    def to_dict(self) -> dict:
        """Convert Output to a dictionary."""
        return {
            "data": {
                key: value.to_dict() if isinstance(value, Data) else value
                for key, value in self.data.items()
            }
        }
