"""Stage base classes."""

from __future__ import annotations

import inspect
import json
from abc import ABC
from pathlib import Path
from typing import Any

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, computed_field

from onemod.config import StageConfig, GroupedConfig, CrossedConfig, ModelConfig
from onemod.utils.parameters import create_params, get_params
from onemod.utils.subsets import create_subsets, get_subset


class Stage(BaseModel, ABC):
    """Stage base class."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    config: StageConfig
    _directory: Path | None = None  # set by Pipeline.add_stage
    _skip_if: set[str] = set()  # defined by class

    @computed_field
    @property
    def type(self) -> str:
        return type(self).__name__

    @computed_field
    @property
    def module(self) -> str:
        return inspect.getfile(self.__class__)

    @property
    def directory(self) -> Path:
        if self._directory is None:
            raise ValueError(f"{self.name} directory has not been set")
        return self._directory

    @directory.setter
    def directory(self, directory: Path) -> None:
        if not directory.exists():
            directory.mkdir()
        self._directory = directory

    @property
    def skip_if(self) -> set[str]:
        return self._skip_if

    @classmethod
    def from_json(cls, filepath: Path | str, name: str) -> Stage:
        """Create stage object from JSON file."""
        with open(filepath, "r") as f:
            pipeline_json = json.load(f)
        stage = cls(**pipeline_json["stages"][name])
        stage.directory = pipeline_json["directory"]
        return stage

    def to_json(self, filepath: Path | str | None = None) -> None:
        """Save stage object as JSON file."""
        filepath = filepath or self.directory.parent / (self.name + ".json")
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def evaluate(
        cls,
        filepath: Path | str,
        name: str,
        method: str = "run",
        *args,
        **kwargs,
    ) -> None:
        """Evaluate stage method.

        Notes
        -----
        * Method designed to be called from the command line
        * Creates stage instance and calls stage method

        """
        stage = cls.from_json(filepath, name)
        if method in stage.skip_if:
            raise ValueError(f"invalid method: {method}")
        stage.__getattribute__(method)(*args, **kwargs)

    def run(self) -> None:
        """Run stage."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.type}({self.name})"


class GroupedStage(Stage, ABC):
    """Grouped stage base class.

    Notes
    -----
    * Any stage that uses the `groupby` setting
    * If you don't want to collect submodel results after stage is run,
      don't implement `collect` method and add 'collect' to `skip_if`

    """

    config: GroupedConfig
    groupby: set[str] = set()
    _subset_ids: set[int] = set()  # set by Stage.create_stage_subsets

    @property
    def subset_ids(self) -> set[int]:
        return self._subset_ids

    def create_stage_subsets(self, data: DataFrame) -> None:
        """Create stage data subsets from groupby."""
        subsets = create_subsets(self.groupby, data)
        if subsets is not None:
            self._subset_ids = list(subsets["subset_id"])
            subsets.to_csv(self.directory / "subsets.csv", index=False)

    def get_stage_subset(self, subset_id: int) -> DataFrame:
        """Get stage data subset."""
        return get_subset(
            self.config.data, self.directory / "subsets.csv", subset_id
        )

    @classmethod
    def evaluate(
        cls,
        filepath: Path | str,
        name: str,
        method: str = "run",
        subset_id: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Evaluate stage method."""
        stage = cls.from_json(filepath, name)
        if method in stage.skip_if:
            raise ValueError(f"invalid method: {method}")
        stage.__getattribute__(method)(subset_id, *args, **kwargs)

    def run(self, subset_id: int) -> None:
        """Run stage submodel."""
        raise NotImplementedError()

    def collect(self) -> None:
        """Collect stage submodel results."""
        raise NotImplementedError()


class CrossedStage(Stage, ABC):
    """Crossed stage base class.

    Notes
    -----
    * Any stage that uses the `crossby` setting
    * If you don't want to collect submodel results after stage is run,
      don't implement `collect` method and add 'collect' to `skip_if`

    """

    config: CrossedConfig
    crossby: set[str] = set()  # set by Stage.create_stage_params
    _param_ids: set[int] = set()  # set by Stage.create_stage_params

    @property
    def param_ids(self) -> set[int]:
        return self._param_ids

    def create_stage_params(self) -> None:
        """Create stage parameter sets from crossby."""
        params = create_params(self.config)
        if params is not None:
            self.crossby = params.drop(columns="param_id").columns
            self._param_ids = params["param_id"]
            params.to_csv(self.directory / "params.csv", index=False)

    def set_params(self, param_id: int) -> Any:
        """Set stage parameters."""
        params = get_params(self.directory / "parameters.csv", param_id)
        for param_name, param_value in params.items():
            self.config[param_name] = param_value

    @classmethod
    def evaluate(
        cls,
        filepath: Path | str,
        name: str,
        method: str = "run",
        param_id: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Evaluate stage method."""
        stage = cls.from_json(filepath, name)
        if method in stage.skip_if:
            raise ValueError(f"invalid method: {method}")
        stage.__getattribute__(method)(param_id, *args, **kwargs)

    def run(self, param_id: int) -> None:
        """Run stage submodel."""
        raise NotImplementedError()

    def collect(self) -> None:
        """Collect stage submodel results."""
        raise NotImplementedError()


class ModelStage(GroupedStage, CrossedStage, ABC):
    """Model stage base class.

    Notes
    -----
    * Models that don't have any `groupby`/`crossby` settings and/or no
      `collect` method should be handled differently (e.g., can skip
      some steps, should save results differently)

    """

    config: ModelConfig

    @classmethod
    def evaluate(
        cls,
        filepath: Path | str,
        name: str,
        method: str = "run",
        subset_id: int | None = None,
        param_id: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        "Evaluate stage method."
        stage = cls.from_json(filepath, name)
        if method in stage.skip_if:
            raise ValueError(f"invalid method: {method}")
        stage.__getattribute__(method)(subset_id, param_id, *args, **kwargs)

    def run(self, subset_id: int | None, param_id: int | None) -> None:
        """Run stage submodel."""
        raise NotImplementedError()

    def fit(self, subset_id: int | None, param_id: int | None) -> None:
        """Fit stage submodel."""
        raise NotImplementedError()

    def predict(self, subset_id: int | None, param_id: int | None) -> None:
        """Predict stage submodel."""
        raise NotImplementedError()
