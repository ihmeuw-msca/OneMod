"""Input/output classes.

Notes
-----
* Classes to organize stage input and output
* Data class is just a placeholder for new onemod data types
* Input and output treated like a dictionary
* Provides validation
* No need to create stage-specific subclasses

"""

from abc import ABC
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_serializer, validate_call


class Data(BaseModel):
    """Dummy data class."""

    stage: str
    path: Path


class IO(BaseModel, ABC):
    """Stage input/output base class."""

    model_config = ConfigDict(frozen=True)

    stage: str
    items: dict[str, Path | Data] = {}

    @model_serializer
    def serialize_io(self) -> dict[str, Path | Data] | None:
        if self.items:
            return self.items
        return None

    def get(self, key: str, default: Any = None) -> Any:
        if not self.__contains__(key):
            return default
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> Path | Data:
        raise NotImplementedError()

    def __contains__(self, key: str) -> bool:
        return key in self.items


class Input(IO):
    """Stage input class."""

    required: set[str] = set()  # defined by stage class
    optional: set[str] = set()  # defined by stage class
    _expected_names: set[str]
    _expected_types: dict[str, str]

    @property
    def dependencies(self) -> set[str] | None:
        return set(
            item.stage for item in self.items.values() if isinstance(item, Data)
        )

    def model_post_init(self, *args, **kwargs) -> None:
        self._expected_names = {
            item.split(".")[0] for item in {*self.required, *self.optional}
        }
        self._expected_types = {}
        for item in {*self.required, *self.optional}:
            item_name, item_type = item.split(".")
            self._expected_types[item_name] = item_type
        if self.items:
            self._check_cycles()
            self._check_types()

    @validate_call
    def update(self, new_items: dict[str, Path | Data]) -> None:
        self._check_cycles(new_items)
        self._check_types(new_items)
        for item_name, item_value in new_items.items():
            self.items[item_name] = item_value

    def check_missing(self) -> None:
        missing_items = [
            item_name
            for item in self.required
            if (item_name := item.split(".")[0]) not in self.items
        ]
        if missing_items:
            raise KeyError(
                f"{self.stage} missing required input: {missing_items}"
            )

    def _check_cycles(
        self, items: dict[str, Path | Data] | None = None
    ) -> None:
        cycles = []
        items = items or self.items
        for item_name, item_value in items.items():
            try:
                self._check_cycle(item_name, item_value)
            except ValueError:
                cycles.append(item_name)
        if cycles:
            raise ValueError(
                f"Circular dependencies for {self.stage} input: {cycles}"
            )

    def _check_cycle(self, item_name: str, item_value: Path | Data) -> None:
        if isinstance(item_value, Data):
            if item_value.stage == self.stage:
                raise ValueError(
                    f"Circular dependency for {self.stage} input: {item_name}"
                )

    def _check_types(self, items: dict[str, Path | Data] | None = None) -> None:
        invalid_items = []
        items = items or self.items
        for item_name, item_value in items.items():
            try:
                self._check_type(item_name, item_value)
            except TypeError:
                invalid_items.append(item_name)
        if invalid_items:
            raise TypeError(
                f"Invalid types for {self.stage} input: {invalid_items}"
            )

    def _check_type(self, item_name: str, item_value: Path | Data) -> None:
        if item_name in self._expected_types:
            if isinstance(item_value, Data):
                item_value = item_value.path
            if item_value.suffix[1:] != self._expected_types[item_name]:
                raise TypeError(
                    f"Invalid type for {self.stage} input: {item_name}"
                )

    def __getitem__(self, key: str) -> Path | Data:
        if not self.__contains__(key):
            if key not in self._expected_names:
                raise ValueError(f"{self.stage} input {key} has not been set")
            raise KeyError(f"{self.stage} does not contain input '{key}'")
        return self.items[key]

    @validate_call
    def __setitem__(self, item_name: str, item_value: Path | Data) -> None:
        if item_name in self._expected_names:
            self._check_cycle(item_name, item_value)
            self._check_type(item_name, item_value)
            self.items[item_name] = item_value


class Output(IO):
    """Stage output class."""

    items: dict[str, Data] = {}  # defined by stage class

    def __getitem__(self, key: str) -> Data:
        if not self.__contains__(key):
            raise KeyError(f"{self.stage} does not contain output '{key}'")
        return self.items[key]
