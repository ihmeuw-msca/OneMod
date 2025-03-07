"""Input/output classes.

Notes
-----
* Classes to organize stage input and output
* Input and output treated like a dictionary
* Provides validation
* No need to create stage-specific subclasses

"""
# TODO: Use collector with check_cycles and check_types?
# TODO: Error messages weird for check_cycles/check_types?

from abc import ABC
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_serializer

from onemod.dtypes import Data
from onemod.dtypes.unique_sequence import unique_list
from onemod.validation.error_handling import (
    ValidationErrorCollector,
    handle_error,
)


class IO(BaseModel, ABC):
    """Stage input/output base class."""

    model_config = ConfigDict(frozen=True)

    stage: str
    items: dict[str, Data] = {}

    @model_serializer
    def serialize_io(self) -> dict[str, dict[str, Any]]:
        # Simplify output to config files
        input_dict: dict[str, dict[str, Any]] = {}
        for item_name, item_value in self.items.items():
            input_dict[item_name] = item_value.model_dump(
                exclude_none=True, serialize_as_any=True
            )
        return input_dict

    def get(self, item_name: str, default: Any = None) -> Any:
        return self.items.get(item_name, default)

    def __getitem__(self, item_name: str) -> Data:
        return self.items[item_name]

    def __contains__(self, item_name: str) -> bool:
        return item_name in self.items


class Input(IO):
    """Stage input class."""

    required: dict[str, Data] = {}
    optional: dict[str, Data] = {}

    @property
    def dependencies(self) -> list[str]:
        return unique_list(
            [item.stage for item in self.items.values() if item.stage]
        )

    def model_post_init(self, *args, **kwargs) -> None:
        if len(set(self.required).intersection(self.optional)):
            raise ValueError("Required and optional input cannot share keys")
        if self.items:
            # Note: silently removes anything not in required or optional
            for item_name in list(self.items):
                if item_name not in list(self.required) + list(self.optional):
                    del self.items[item_name]
            self._check_cycles()
            self._check_types()

    def update(self, items: dict[str, Data | Path | str]) -> None:
        # Note: silently ignores anything not in required or optional
        data_items = {}
        for item_name, item_value in items.items():
            if item_name in list(self.required) + list(self.optional):
                if isinstance(item_value, Data):
                    data_items[item_name] = item_value
                elif isinstance(item_value, (Path, str)):
                    data_items[item_name] = Input.path_to_data(item_value)
                else:
                    raise TypeError(
                        f"Invalid input item: {item_name}={item_value}"
                    )

        self._check_cycles(data_items)
        self._check_types(data_items)

        for item_name, item_value in data_items.items():
            self.items[item_name] = item_value

    def remove(self, item: str) -> None:
        if item in self.items:
            del self.items[item]

    def clear(self) -> None:
        self.items.clear()

    def check_missing(
        self,
        items: dict[str, Data] | None = None,
        collector: ValidationErrorCollector | None = None,
    ) -> None:
        """Check stage input items have been defined.

        Parameters
        ----------
        items : dict of str: Data, optional
            Input items to check. If None, check all stage input items.
            Default is None.

        Raises
        ------
        KeyError
            If any stage input items have not been defined.

        """
        items = items or self.items
        missing_items = [
            item_name for item_name in self.required if item_name not in items
        ]
        if missing_items:
            handle_error(
                self.stage,
                "Data validation",
                KeyError,
                f"Stage '{self.stage}' missing required input: {missing_items}",
                collector,
            )

    def check_exists(
        self,
        item_names: list[str] | None = None,
        upstream_stages: list[str] | None = None,
        collector: ValidationErrorCollector | None = None,
    ) -> None:
        """Check stage input items exist.

        Parameters
        ----------
        item_names : list of str, optional
            Names of input items to check. If None, check all input
            path items and all input data items in `upstream_stages`.
            Default is None.
        upstream_stages : list of str, optional
            Names of upstream stages to check input items from. If None,
            check input items from all upstream stages.

        Raises
        ------
        FileNotFoundError
            If any stage input items do not exist.

        """
        if item_names is None:
            item_names = list(self.items)
        if upstream_stages is None:
            upstream_stages = self.dependencies

        missing_items: dict[str, str | None] = {}
        for item_name in item_names:
            item_value = self.__getitem__(item_name)
            if item_value.stage in upstream_stages:
                if (item_path := item_value.path) is None:
                    missing_items[item_name] = None
                elif not item_path.exists():
                    missing_items[item_name] = str(item_path)
            else:
                continue

        if missing_items:
            handle_error(
                self.stage,
                "Data validation",
                FileNotFoundError,
                f"Stage '{self.stage}' input items do not exist: {missing_items}",
                collector,
            )

    def _check_cycles(self, items: dict[str, Data] | None = None) -> None:
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

    def _check_cycle(self, item_name: str, item_value: Data) -> None:
        if item_value.stage == self.stage:
            raise ValueError(
                f"Circular dependency for {self.stage} input: {item_name}"
            )

    def _check_types(self, items: dict[str, Data] | None = None) -> None:
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

    def _check_type(self, item_name: str, item_value: Data) -> None:
        if item_name in list(self.required):
            expected = self.required[item_name]
        elif item_name in list(self.optional):
            expected = self.optional[item_name]
        else:
            return

        if item_value.format != expected.format:
            raise TypeError(f"Invalid type for {self.stage} input: {item_name}")

    @staticmethod
    def path_to_data(path: Path | str) -> Data:
        if isinstance(path, str):
            path = Path(path)
        return Data(format=path.suffix[1:] or "directory", path=path)

    def __getitem__(self, item_name: str) -> Data:
        if item_name not in self.items:
            if item_name in list(self.required) + list(self.optional):
                raise ValueError(
                    f"{self.stage} input '{item_name}' has not been set"
                )
            raise KeyError(f"{self.stage} does not contain input '{item_name}'")
        return self.items[item_name]

    def __setitem__(
        self, item_name: str, item_value: Data | Path | str
    ) -> None:
        if item_name in list(self.required) + list(self.optional):
            if isinstance(item_value, (Path, str)):
                item_value = self.path_to_data(item_value)
            self._check_cycle(item_name, item_value)
            self._check_type(item_name, item_value)
            self.items[item_name] = item_value


class Output(IO):
    """Stage output class."""

    directory: Path
    items: dict[str, Data] = {}

    def model_post_init(self, *args, **kwargs) -> None:
        for item_name, item_value in self.items.items():
            item_value.stage = self.stage
            item_value.path = self.directory / (
                item_name
                if (item_format := item_value.format) == "directory"
                else f"{item_name}.{item_format}"
            )

    def check_exists(
        self,
        item_names: list[str] | None = None,
        collector: ValidationErrorCollector | None = None,
    ) -> None:
        # TODO: add method arg
        if item_names is None:
            item_names = list(self.items)

        missing_items: dict[str, str | None] = {}
        for item_name in item_names:
            item_value = self.__getitem__(item_name)
            if (item_path := item_value.path) is None:
                missing_items[item_name] = None
            elif not item_path.exists():
                missing_items[item_name] = str(item_path)

        if missing_items:
            handle_error(
                self.stage,
                "Data validation",
                FileNotFoundError,
                f"Stage '{self.stage}' output items do not exist: {missing_items}",
                collector,
            )

    def __getitem__(self, item_name: str) -> Data:
        if item_name not in self.items:
            raise KeyError(
                f"Stage '{self.stage}' does not contain output '{item_name}'"
            )
        return self.items[item_name]
