"""Stage base classes."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from inspect import getfile
from pathlib import Path
from typing import Any

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, computed_field

import onemod.stage as onemod_stages
from onemod.config import CrossedConfig, GroupedConfig, ModelConfig, StageConfig
from onemod.io import Input, Output
from onemod.types import Data
from onemod.utils.parameters import create_params, get_params
from onemod.utils.subsets import create_subsets, get_subset
from onemod.validation import ValidationErrorCollector, validation_context


class Stage(BaseModel, ABC):
    """Stage base class."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    config: StageConfig
    # input: dict[str, Data] = {}  # also set by Stage.__call__
    # TODO: confirm this is the intended usage
    input: Input
    output: Output
    _directory: Path | None = None  # set by Stage.from_json, Pipeline.add_stage
    _module: Path | None = None  # set by Stage.from_json
    _skip_if: set[str] = set()  # defined by class
    _inputs: set[str] = set()  # defined by class
    _outputs: set[str] = set()  # defined by class

    @computed_field
    @property
    def type(self) -> str:
        return type(self).__name__

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

    @module.setter
    def module(self, module: str | None) -> None:
        self._module = module

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

    @property
    def dependencies(self) -> set[str]:
        return self.input.dependencies

    def to_dict(self) -> dict:
        """Convert stage to dictionary representaion."""
        return {
            "name": self.name,
            "type": self.type,
            "config": self.config.model_dump(),
            "inputs": self.input.to_dict(),
            "outputs": self.output.to_dict(),
            "dependencies": self.dependencies,
        }

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
            directory = Path(config["directory"]) / name
            config = config["stages"][name]
        else:
            directory = Path(filepath).parent
            
        stage = cls(
            name=config["name"],
            type=config["type"],
            config=StageConfig(**config["config"]),
            module=config.get("module", None),
            input=Input.from_dict(config["inputs"]),
            output=Output.from_dict(config["outputs"]),
            dependencies=config.get("dependencies", set()),
        )
        stage.directory = directory
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
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def evaluate(
        cls,
        filepath: Path | str,
        name: str | None = None,
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
        name : str or None, optional
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
        stage = cls.from_json(filepath, name, from_pipeline)
        if method in stage.skip_if:
            raise AttributeError(f"{stage.name} skips the '{method}' method")
        try:
            stage.__getattribute__(method)(*args, **kwargs)
        except AttributeError:
            raise AttributeError(
                f"{stage.name} does not have a '{method}' method"
            )
    
    def validate_inputs(self, collector: ValidationErrorCollector) -> None:
        """Validate stage inputs."""
        self.input.validate()

        for key, value in self.input.items():
            if isinstance(value, Data):
                value.validate_data(collector)
        
    def validate_outputs(self, collector: ValidationErrorCollector) -> None:
        """Validate stage outputs."""
        for key, value in self.output.items():
            if isinstance(value, Data):
                value.validate_data(collector)
        
    def run(self, subset_id: int | None = None, param_id: int | None = None) -> None:
        """User-defined method to run stage."""
        raise NotImplementedError("Subclasses must implement this method.")
        
    def execute(self, subset_id: int | None = None, param_id: int | None = None) -> None:
        """Execute stage."""
        with validation_context() as collector:
            self.validate_inputs(collector)
            
            if collector.errors:
                errors = collector.get_errors()
                raise ValueError(f"Validation failed for stage {self.name} inputs: {errors}")
        
        self.run(subset_id, param_id)
        
        with validation_context() as collector:
            self.validate_outputs(collector)
            
            if collector.errors:
                errors = collector.get_errors()
                raise ValueError(f"Validation failed for stage {self.name} outputs: {errors}")

    def __call__(self, **input: Data) -> None:
        """Define stage dependencies."""
        # TODO: Validation
        self.input = input

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

    def execute(self, subset_id: int) -> None:
        """Run stage submodel."""
        self.validate_inputs()
        
        if subset_id is None:
            for subset in self._subset_ids:
                self.run(subset)
        else:
            self.run(subset_id)
        
        self.validate_outputs()

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

    def execute(self, param_id: int) -> None:
        """Run stage submodel."""
        self.validate_inputs()
        
        if param_id is None:
            for param in self._param_ids:
                self.set_params(param)
                self.run(param)
        else:
            self.set_params(param_id)
            self.run(param_id)
        
        self.validate_outputs()

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

    def execute(self, subset_id: int | None, param_id: int | None) -> None:
        """Execute stage submodel."""
        self.validate_inputs()
        
        self.run(subset_id, param_id)
        # if subset_id is None:
        #     for subset in self._subset_ids:
        #         if param_id is None:
        #             for param in self._param_ids:
        #                 self.fit(subset, param)
        #                 self.predict(subset, param)
        #         else:
        #             self.fit(subset, param_id)
        #             self.predict(subset, param_id)
        # else:
        #     if param_id is None:
        #         for param in self._param_ids:
        #             self.fit(subset_id, param)
        #             self.predict(subset_id, param)
        #     else:
        #         self.fit(subset_id, param_id)
        #         self.predict(subset_id, param_id)
                
        self.validate_outputs()

    @abstractmethod
    def fit(self, subset_id: int | None, param_id: int | None) -> None:
        """Fit stage submodel."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def predict(self, subset_id: int | None, param_id: int | None) -> None:
        """Predict stage submodel."""
        raise NotImplementedError("Subclasses must implement this method.")
