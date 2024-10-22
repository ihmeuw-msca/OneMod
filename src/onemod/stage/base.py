"""Stage base classes."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from inspect import getfile
from pathlib import Path
from typing import Literal

from pandas import DataFrame
from pplkit.data.interface import DataInterface
from pydantic import BaseModel, ConfigDict, Field, computed_field, validate_call

import onemod.stage as onemod_stages
from onemod.config import ModelConfig, StageConfig
from onemod.io import Input, Output
from onemod.dtypes import Data
from onemod.serializers.functions import load_config
from onemod.utils.parameters import create_params, get_params
from onemod.utils.subsets import create_subsets, get_subset
from onemod.validation import ValidationErrorCollector, handle_error


class Stage(BaseModel, ABC):
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
    _dataif: DataInterface | None = None  # set by Pipeline.build, from_json
    _module: Path | None = None  # set by from_json
    _skip: set[str] = set()  # defined by class
    _input: Input | None = None  # set by __call__, from_json
    _required_input: set[str] = set()  # name.extension, defined by class
    _optional_input: set[str] = set()  # name.extension, defined by class
    _output: set[str] = set()  # name.extension, defined by class

    @property
    def dataif(self) -> DataInterface:
        if self._dataif is None:
            raise AttributeError(f"{self.name} dataif has not been set")
        return self._dataif

    def set_dataif(self, config_path: Path | str) -> None:
        directory = Path(config_path).parent
        self._dataif = DataInterface(
            directory=directory,
            config=config_path,
            output=directory / self.name,
        )
        for item_name, item_value in self.input.items.items():
            if isinstance(item_value, Path):
                self._dataif.add_dir(item_name, item_value)
            elif isinstance(item_value, Data):
                self._dataif.add_dir(
                    item_name, directory / item_value.stage / item_value.path
                )
        if not (directory / self.name).exists():
            (directory / self.name).mkdir()

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
            output_items[item_name] = Data(stage=self.name, path=item)
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

        """
        try:
            config = load_config(config_path)["stages"][stage_name]
        except KeyError:
            f"Config does not contain a stage named '{stage_name}'"
        stage = cls(**config)
        stage.set_dataif(config_path)
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
    groupby : set of str or None, optional
        Column names used to create data subsets.
        Default is None.
    input_validation : dict, optional
        Description.
    output_validation : dict, optional
        Description.

    """

    config: ModelConfig
    groupby: set[str] | None = None
    _crossby: set[str] | None = None  # set by Stage.create_stage_params
    _subset_ids: set[int] = set()  # set by Stage.create_stage_subsets
    _param_ids: set[int] = set()  # set by Stage.create_stage_params
    _required_input: set[str] = set()  # data required for groupby

    @computed_field
    @property
    def crossby(self) -> set[str]:
        return self._crossby

    @property
    def subset_ids(self) -> set[int]:
        if self.groupby is not None and not self._subset_ids:
            try:
                subsets = self.dataif.load_output("subsets.csv")
                self._subset_ids = set(subsets["subset_id"])
            except FileNotFoundError:
                raise AttributeError(
                    f"{self.name} data subsets have not been created"
                )
        return self._subset_ids

    @property
    def param_ids(self) -> set[int]:
        if self.crossby is not None and not self._param_ids:
            try:
                params = self.dataif.load_output("parameters.csv")
                self._param_ids = set(params["param_id"])
            except FileNotFoundError:
                raise AttributeError(
                    f"{self.name} parameter sets have not been created"
                )
        return self._param_ids

    def create_stage_subsets(self, data: Path | str) -> None:
        """Create stage data subsets from groupby."""
        subsets = create_subsets(
            self.groupby, self.dataif.load(data, columns=self.groupby)
        )
        self._subset_ids = set(subsets["subset_id"])
        self.dataif.dump_output(subsets, "subsets.csv", index=False)

    def get_stage_subset(self, subset_id: int) -> DataFrame:
        """Get stage data subset."""
        return get_subset(
            self.dataif.load_data(),
            self.dataif.load_output("subsets.csv"),
            subset_id,
        )

    def create_stage_params(self) -> None:
        """Create stage parameter sets from config."""
        params = create_params(self.config)
        if params is not None:
            self._crossby = set(params.drop(columns="param_id").columns)
            self._param_ids = set(params["param_id"])
            self.dataif.dump_output("parameters.csv", index=False)

    def set_params(self, param_id: int) -> None:
        """Set stage parameters."""
        params = get_params(self.dataif.load_output("parameters.csv"), param_id)
        for param_name, param_value in params.items():
            self.config[param_name] = param_value

    def get_pipeline_groupby(self) -> list[str] | None:
        """Get pipeline groupby attribute.

        TODO: Make more generalized function to get fields from config

        """
        # Using cached load_config instead of self.dataif.load_config
        return load_config(self.dataif.config).get("groupby")

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
        if self.groupby is not None:
            stage_str += f", groupby={self.groupby}"
        if self.crossby is not None:
            stage_str += f", crossby={self.crossby}"
        return stage_str + ")"
