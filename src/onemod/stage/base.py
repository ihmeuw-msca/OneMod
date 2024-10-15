"""Stage base classes."""

from __future__ import annotations

import json
from abc import ABC
from functools import cached_property
from inspect import getfile
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, computed_field, validate_call

import onemod.stage as onemod_stages
from onemod.config import CrossedConfig, GroupedConfig, ModelConfig, StageConfig
from onemod.io import Data, Input, Output
from onemod.utils.parameters import create_params, get_params
from onemod.utils.subsets import create_subsets, get_subset


class Stage(BaseModel, ABC):
    """Stage base class."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    config: StageConfig
    _pipeline: str  # set by Stage.from_json
    _directory: Path | None = None  # set by Pipeline.add_stage, Stage.from_json
    _module: Path | None = None  # set by Stage.from_json
    _skip_if: set[str] = set()  # defined by class
    _input: Input | None = None  # set by Stage.__call__, Stage.from_json
    _required_input: set[str] = set()  # name.extension, defined by class
    _optional_input: set[str] = set()  # name.extension, defined by class
    _output: set[str] = set()  # name.extension, defined by class

    @property
    def pipeline(self) -> str:
        if self._pipeline is None:
            raise AttributeError(f"{self.name} pipeline has not been set")
        return self._pipeline

    @property
    def directory(self) -> Path:
        if self._directory is None:
            raise AttributeError(f"{self.name} directory has not been set")
        return self._directory

    @directory.setter
    def directory(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        if not self._directory.exists():
            self._directory.mkdir()

    @computed_field
    @property
    def module(self) -> str | None:
        if self._module is None and not hasattr(
            onemod_stages, self.type
        ):  # custom stage
            try:
                return getfile(self.__class__)
            except TypeError:
                raise TypeError(f"Could not find module for {self.name} stage")
        return self._module

    @property
    def skip_if(self) -> set[str]:
        return self._skip_if

    @computed_field
    @property
    def input(self) -> Input | None:
        if self._input is None:
            raise AttributeError(f"{self.name} input has not been set")
        return self._input

    @cached_property
    def output(self) -> Output:
        output_items = {}
        for item in self._output:
            item_name = item.split(".")[0]  # remove extension
            output_items[item_name] = Data(
                stage=self.name, path=self.directory / item
            )
        return Output(stage=self.name, items=output_items)

    @property
    def dependencies(self) -> set[str]:
        if self.input is None:
            return set()
        return self.input.dependencies

    @computed_field
    @property
    def type(self) -> str:
        return type(self).__name__

    @classmethod
    def from_json(
        cls,
        filepath: Path | str,
        name: str | None = None,
        from_pipeline: bool = False,
    ) -> Stage:
        """Load stage from JSON file.

        Parameters
        ----------
        filepath : Path or str
            Path to config file.
        name : str or None, optional
            Stage name, required if `from_pipeline` is True.
            Default is None.
        from_pipeline : bool, optional
            Whether `filepath` is a pipeline or stage config file.
            Default is False.

        Returns
        -------
        Stage
            Stage instance.

        Notes
        -----
        If `from_pipeline` is True, the stage directory is set to
        pipeline.directory / `name`. Otherwise, the stage directory is
        set to the parent directory of `filepath`.

        """
        with open(filepath, "r") as f:
            config = json.load(f)
        if from_pipeline:
            pipeline = config["name"]
            directory = Path(config["directory"]) / name
            config = config["stages"][name]
        else:
            pipeline = None
            directory = Path(filepath).parent
        stage = cls(**config)
        stage._pipeline = pipeline
        stage.directory = directory
        if "module" in config:
            stage._module = config["module"]
        if "input" in config:
            stage(**config["input"])
        return stage

    def to_json(self, filepath: Path | str | None = None) -> None:
        """Save stage as JSON file.

        Parameters
        ----------
        filepath : Path, str or None, optional
            Where to save the config file. If None, file is saved at
            stage.directory / (stage.name + ."json").
            Default is None.

        """
        filepath = filepath or self.directory / (self.name + ".json")
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4, exclude_none=True))

    @classmethod
    def evaluate(
        cls,
        filepath: Path | str,
        stage_name: str | None = None,
        from_pipeline: bool = False,
        method: str = "run",
        *args,
        **kwargs,
    ) -> None:
        """Evaluate stage method.

        Parameters
        ----------
        filepath : Path or str
            Path to config file.
        stage_name : str or None, optional
            Stage name, required if `from_pipeline` is True.
            Default is None.
        from_pipeline : bool, optional
            Whether `filepath` is a pipeline or stage config file.
            Default is False.
        method : str, optional
            Name of stage method to evaluate. Default is 'run'.

        Notes
        -----
        This class method is designed to be called from the command
        line. It creates a stage instance from the config file, then
        calls the stage method.

        """
        stage = cls.from_json(filepath, stage_name, from_pipeline)
        if method in stage.skip_if:
            raise AttributeError(f"{stage.name} skips the '{method}' method")
        try:
            stage.__getattribute__(method)(*args, **kwargs)
        except AttributeError:
            raise AttributeError(
                f"{stage.name} does not have a '{method}' method"
            )

    def run(self) -> None:
        """Run stage."""
        raise NotImplementedError()

    @validate_call
    def __call__(self, **input: Data | Path) -> None:
        """Define stage dependencies."""
        if self._input is None:
            self._input = Input(
                stage=self.name,
                required=self._required_input,
                optional=self._optional_input,
            )
        self.input.check_missing({**self.input.items, **input})
        self.input.update(input)

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
    _required_input: set[str] = {"data"}

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
        if isinstance(self.input["data"], Path):
            data_path = self.input["data"]
        else:
            data_path = self.input["data"].path
        return get_subset(data_path, self.directory / "subsets.csv", subset_id)

    @classmethod
    def evaluate(
        cls,
        filepath: Path | str,
        stage_name: str,
        method: str = "run",
        subset_id: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Evaluate stage method."""
        stage = cls.from_json(filepath, stage_name)
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
        stage_name: str,
        method: str = "run",
        param_id: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Evaluate stage method."""
        stage = cls.from_json(filepath, stage_name)
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
        stage_name: str,
        method: str = "run",
        subset_id: int | None = None,
        param_id: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        "Evaluate stage method."
        stage = cls.from_json(filepath, stage_name)
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
