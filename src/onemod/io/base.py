"""Input/output classes.

Notes
-----
* Classes to organize stage input and output
* Input and output treated like a dictionary
* Provides validation
* No need to create stage-specific subclasses

"""

from abc import ABC
from pathlib import Path
from typing import Any, Optional

from pydantic import ConfigDict, model_serializer, validate_call

from onemod.base_models import SerializableModel
from onemod.types import Data
from onemod.validation.error_handling import ValidationErrorCollector


class IO(SerializableModel, ABC):
    """Stage input/output base class."""

    model_config = ConfigDict(frozen=True)

    stage: str
    items: dict[str, Data | Path] = {}

    @model_serializer
    def serialize_io(self) -> dict[str, str | dict[str, str]] | None:
        # Simplify output to config files
        if self.items:
            input_dict = {}
            for item_name, item_value in self.items.items():
                if isinstance(item_value, Path):
                    input_dict[item_name] = str(item_value)
                elif isinstance(item_value, Data):
                    input_dict[item_name] = item_value.model_dump()
            return input_dict
        return None
    
    def get(self, item_name: str, default: Any = None) -> Any:
        return self.items.get(item_name, default)

    def __getitem__(self, item_name: str) -> Data | Path:
        return self.items[item_name]

    def __contains__(self, item_name: str) -> bool:
        return item_name in self.items


class Input(IO):
    """Stage input class."""

    required: set[str] = set()  # name.extension, defined by stage class
    optional: set[str] = set()  # name.extension, defined by stage class
    _expected_names: set[str]
    _expected_types: dict[str, str]

    @property
    def dependencies(self) -> set[str]:
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
            for item_name in list(self.items.keys()):
                if item_name not in self._expected_names:
                    del self.items[item_name]
            self._check_cycles()
            self._check_types()

    @validate_call
    def update(self, items: dict[str, Data | Path]) -> None:
        self._check_cycles(items)
        self._check_types(items)
        for item_name, item_value in items.items():
            if item_name in self._expected_names:
                self.items[item_name] = item_value

    def remove(self, item: str) -> None:
        if item in self.items:
            del self.items[item]

    def clear(self) -> None:
        self.items.clear()

    def check_missing(
        self, items: dict[str, Data | Path] | None = None
    ) -> None:
        items = items or self.items
        missing_items = [
            item_name
            for item in self.required
            if (item_name := item.split(".")[0]) not in items
        ]
        if missing_items:
            raise KeyError(
                f"{self.stage} missing required input: {missing_items}"
            )

    def _check_cycles(
        self, items: dict[str, Data | Path] | None = None
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

    def _check_cycle(self, item_name: str, item_value: Data | Path) -> None:
        if isinstance(item_value, Data):
            if item_value.stage == self.stage:
                raise ValueError(
                    f"Circular dependency for {self.stage} input: {item_name}"
                )

    def _check_types(self, items: dict[str, Data | Path] | None = None) -> None:
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

    def _check_type(self, item_name: str, item_value: Data | Path) -> None:
        if item_name in self._expected_types:
            if isinstance(item_value, Data):
                item_value = item_value.path
            if item_value.suffix[1:] != self._expected_types[item_name]:
                raise TypeError(
                    f"Invalid type for {self.stage} input: {item_name}"
                )

    def __getitem__(self, item_name: str) -> Data | Path:
        if item_name not in self.items:
            if item_name in self._expected_names:
                raise ValueError(
                    f"{self.stage} input '{item_name}' has not been set"
                )
            raise KeyError(f"{self.stage} does not contain input '{item_name}'")
        return self.items[item_name]

    @validate_call
    def __setitem__(self, item_name: str, item_value: Data | Path) -> None:
        if item_name in self._expected_names:
            self._check_cycle(item_name, item_value)
            self._check_type(item_name, item_value)
            self.items[item_name] = item_value


class Output(IO):
    """Stage output class."""

    items: dict[str, Data] = {}  # defined by stage class
    _expected_names: set[str]

    def __getitem__(self, item_name: str) -> Data:
        if item_name not in self.items:
            raise KeyError(
                f"{self.stage} does not contain output '{item_name}'"
            )
        return self.items[item_name]
