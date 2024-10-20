"""Stage base classes."""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from inspect import getfile
from pathlib import Path
from typing import Literal

import pandas as pd
from pandas import DataFrame
from pydantic import ConfigDict, Field, computed_field, validate_call

import onemod.stage as onemod_stages
from onemod.base_models import SerializableModel
from onemod.config import ModelConfig, StageConfig
from onemod.io import Input, Output
from onemod.dtypes import Data
from onemod.utils.parameters import create_params, get_params
from onemod.utils.subsets import create_subsets, get_subset
from onemod.validation import ValidationErrorCollector, handle_error


class Stage(SerializableModel, ABC):
    """Stage base class.

    Parameters
    ----------
    name : str
        Stage name.
    config : StageConfig
        Stage configuration.
    input_validation : dict, optional
        Description.
    output_validation : dict, optional
        Description.

    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    config: StageConfig
    input_validation: dict[str, Data] = Field(default_factory=dict)
    output_validation: dict[str, Data] = Field(default_factory=dict)
    _pipeline: str  # set by Stage.from_json
    _directory: Path | None = None  # set by Pipeline.add_stage, Stage.from_json
    _module: Path | None = None  # set by Stage.from_json
    _skip: set[str] = set()  # defined by class
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
    def skip(self) -> set[str]:
        return self._skip

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
    def from_json(cls, config_path: Path | str, stage_name: str) -> Stage:
        """Load stage from JSON file.

        Parameters
        ----------
        config_path : Path or str
            Path to config file.
        stage_name : str
            Stage name.

        Returns
        -------
        Stage
            Stage instance.

        Notes
        -----
        Stage directory set to pipeline.directory / `stage_name`.

        """
        with open(config_path, "r") as f:
            config = json.load(f)
        pipeline = config["name"]
        directory = Path(config["directory"]) / stage_name
        if stage_name not in config["stages"]:
            raise KeyError(
                f"Config does not contain a stage named '{stage_name}'"
            )
        config = config["stages"][stage_name]
        stage = cls(**config)
        stage._pipeline = pipeline
        stage.directory = directory
        if "module" in config:
            stage._module = config["module"]
        if "crossby" in config:
            stage._crossby = config["crossby"]
        if "input" in config:
            stage(**config["input"])
        return stage

    @validate_call
    def evaluate(
        self,
        method: Literal["run", "fit", "predict"] = "run",
        backend: Literal["local", "jobmon"] = "local",
        **kwargs,
    ) -> None:
        """Evaluate stage method.

        Parameters
        ----------
        method : str, optional
            Name of method to evaluate. Default is 'run'.
        backend : str, optional
            Whether to evaluate the method locally or with Jobmon.
            Default is 'local'.

        Other Parameters
        ----------------
        config : Path or str, optional
            Path to config file. Required if `backend` is 'jobmon'.
        cluster : str, optional
            Cluster name. Required if `backend` is 'jobmon'.
        resources : Path or str, optional
            Path to resources yaml file. Required if `backend` is
            'jobmon'.

        Notes
        -----
        If stage does not implement `method`, and `method` not in
        `skip`, the `run` method will be evaluated. For example, a data
        preprocessing stage might implement `run`, skip `predict`, and
        evaluate `run` when `fit` is called. Alternatively, a plotting
        stage might implement `run`, skip `fit`, and evaluate `run` when
        `predict` is called.

        """
        if method in self.skip:
            warnings.warn(f"{self.name} skips the '{method}' method")
            return
        method = method if hasattr(self, method) else "run"
        if backend == "jobmon":
            from onemod.backend import evaluate_with_jobmon

            evaluate_with_jobmon(model=self, method=method, **kwargs)
        else:
            self.__getattribute__(method)()

    def validate_build(self, collector: ValidationErrorCollector) -> None:
        """Perfom build-time validation."""
        if self.input_validation:
            for item_name, schema in self.input_validation.items():
                if isinstance(schema, Data):
                    schema.validate_metadata(kind="input", collector=collector)

        if self.output_validation:
            for item_name, schema in self.output_validation.items():
                if isinstance(schema, Data):
                    schema.validate_metadata(kind="output", collector=collector)

    def validate_run(self, collector: ValidationErrorCollector) -> None:
        """Perfom run-time validation."""
        if self.input_validation:
            for item_name, schema in self.input_validation.items():
                data_path = self.input.get(item_name)
                if data_path:
                    schema.path = Path(data_path)
                    schema.validate_data(collector)
                else:
                    handle_error(
                        self.name,
                        "Input validation",
                        ValueError,
                        f"Input data path for '{item_name}' not found in stage inputs.",
                        collector,
                    )

    def validate_outputs(self, collector: ValidationErrorCollector) -> None:
        """Perform post-run validation of outputs."""
        if self.output_validation:
            for item_name, data_spec in self.output_validation.items():
                data_output = self.output.get(item_name)
                if data_output:
                    data_spec.path = Path(data_output.path)
                    data_spec.validate_data(collector)
                else:
                    handle_error(
                        self.name,
                        "Output Validation",
                        KeyError,
                        f"Output data '{item_name}' not found after stage execution.",
                        collector,
                    )

    @abstractmethod
    def run(self) -> None:
        """Run stage."""
        raise NotImplementedError("Subclasses must implement this method.")

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


class ModelStage(Stage, ABC):
    """Model stage base class.

    Model stages can be run separately for data subsets using the
    `groupby` attribute. For example, a single stage could have separate
    models by sex_id or age_group_id.

    Model stages can also be run for different parameter combinations
    using the `crossby` attribute. For example, a single stage could be
    run for various hyperparameter values, and then the results could be
    combined into an ensemble. Any parameter in config.crossable_params
    can be specified as either a single value or a list of values.

    When a model stage method is evaluated, all submodels (identified by
    their `subset_id` and `param_id`) are evaluated, and then the
    submodel results are collected using the `collect` method.

    Parameters
    ----------
    name : str
        Stage name.
    config : ModelConfig
        Stage configuration.
    groupby : set of str, optional
        Column names used to create data subsets.
        Default is an empty set.
    input_validation : dict, optional
        Description.
    output_validation : dict, optional
        Description.

    """

    config: ModelConfig
    groupby: set[str] = set()
    _crossby: set[str] = set()  # set by Stage.create_stage_params
    _subset_ids: set[int] = set()  # set by Stage.create_stage_subsets
    _param_ids: set[int] = set()  # set by Stage.create_stage_params
    _required_input: set[str] = set()  # 'data' required for `groupby`

    @computed_field
    @property
    def crossby(self) -> set[str]:
        return self._crossby

    @property
    def subset_ids(self) -> set[int]:
        if self.groupby and not self._subset_ids:
            try:
                subsets = pd.read_csv(self.directory / "subsets.csv")
                self._subset_ids = set(subsets["subset_id"])
            except FileNotFoundError:
                raise AttributeError(
                    f"{self.name} data subsets have not been created"
                )
        return self._subset_ids

    @property
    def param_ids(self) -> set[int]:
        if self.crossby and not self._param_ids:
            try:
                params = pd.read_csv(self.directory / "parameters.csv")
                self._param_ids = set(params["param_id"])
            except FileNotFoundError:
                raise AttributeError(
                    f"{self.name} parameter sets have not been created"
                )
        return self._param_ids

    def create_stage_subsets(self, data: DataFrame) -> None:
        """Create stage data subsets from groupby."""
        subsets = create_subsets(self.groupby, data)
        if subsets is not None:
            self._subset_ids = set(subsets["subset_id"])
            subsets.to_csv(self.directory / "subsets.csv", index=False)

    def get_stage_subset(self, subset_id: int) -> DataFrame:
        """Get stage data subset."""
        if isinstance(self.input["data"], Path):
            data_path = self.input["data"]
        else:
            data_path = self.input["data"].path
        return get_subset(data_path, self.directory / "subsets.csv", subset_id)

    def create_stage_params(self) -> None:
        """Create stage parameter sets from config."""
        params = create_params(self.config)
        if params is not None:
            self._crossby = set(params.drop(columns="param_id").columns)
            self._param_ids = set(params["param_id"])
            params.to_csv(self.directory / "parameters.csv", index=False)

    def set_params(self, param_id: int) -> None:
        """Set stage parameters."""
        params = get_params(self.directory / "parameters.csv", param_id)
        for param_name, param_value in params.items():
            self.config[param_name] = param_value

    @validate_call
    def evaluate(
        self,
        method: Literal["run", "fit", "predict", "collect"] = "run",
        backend: Literal["local", "jobmon"] = "local",
        **kwargs,
    ) -> None:
        """Evaluate model stage method.

        Parameters
        ----------
        method : str, optional
            Name of method to evaluate. Default is 'run'.
        backend : str, optional
            Whether to evaluate the method locally or with Jobmon.
            Default is 'local'.

        Other Parameters
        ----------------
        subset_id : int, optional
            Submodel data subset ID. Ignored if `backend` is 'jobmon'.
        param_id : int, optional
            Submodel parameter set ID. Ignored if `backend` is 'jobmon'.
        config : Path or str, optional
            Path to config file. Required if `backend` is 'jobmon'.
        cluster : str, optional
            Cluster name. Required if `backend` is 'jobmon'.
        resources : Path or str, optional
            Path to resources yaml file. Required if `backend` is
            'jobmon'.

        Notes
        -----
        If either `subset_id` or `param_id` is passed, `method` will be
        evaluated for the corresponding submodel (unless `backend` is
        'jobmon' or `method` is 'collect', in which case `subset_id`
        and `param_id` are ignored). Otherwise, `method` will be
        evaluated for all submodels, and then `collect` will be
        evaluated to collect the submodel results.

        """
        if method in self.skip:
            warnings.warn(f"{self.name} skips the '{method}' method")
            return
        if backend == "jobmon":
            from onemod.backend import evaluate_with_jobmon

            evaluate_with_jobmon(model=self, method=method, **kwargs)
        else:
            if method == "collect":
                self.collect()
            else:
                subset_id = kwargs.get("subset_id")
                param_id = kwargs.get("param_id")
                if subset_id is not None or param_id is not None:
                    self.__getattribute__(method)(subset_id, param_id)
                else:
                    from onemod.backend import evaluate_local

                    evaluate_local(model=self, method=method)

    @abstractmethod
    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run stage submodel."""
        raise NotImplementedError("Subclasses must implement this method.")

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit stage submodel."""
        raise NotImplementedError(
            "Subclasses must implement this method if not skipped."
        )

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Predict stage submodel."""
        raise NotImplementedError(
            "Subclasses must implement this method if not skipped."
        )

    @abstractmethod
    def collect(self) -> None:
        """Collect stage submodel results."""
        raise NotImplementedError("Subclasses must implement this method.")

    def __repr__(self) -> str:
        stage_str = f"{self.type}({self.name}"
        if self.grouby:
            stage_str += f", groupby={self.groupby}"
        if self.crossby:
            stage_str += f", crossby={self.crossby}"
        return stage_str + ")"
