"""Stage base classes."""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from inspect import getfile
from pathlib import Path
from typing import Any, Literal

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field, validate_call

import onemod.stage as onemod_stages
from onemod.config import StageConfig
from onemod.dtypes import Data
from onemod.fsutils import DataInterface
from onemod.io import Input, Output
from onemod.utils.decorators import computed_property
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
        Optional specification of input data validation.
    output_validation : dict, optional
        Optional specification of output data validation.

    Notes
    -----
    * Private attributes that are defined automatically:
      * `_dataif : `DataInterface` object for loading/dumping input,
        created in `Pipeline.build()` or `Stage.from_json()`.
      * `_module`: Path to custom stage definition, created in
        `Stage.module` or `Stage.from_json()`.
      * `_input`: `Input` object that organizes `Stage` input, created
        in `Stage.input` or `Stage.from_json()`, modified by
        `Stage.__call__()`.
    * Private attributes that must be defined by class:
      * `_required_input`, `_optional_input`, `_output`: Strings with
        syntax "f{name}.{extension}". For example, "data.parquet". If
        input/output is a directory instead of a file, exclude the
        extension. For example, "submodels".
      * `_skip`: Methods that the stage does not implement (e.g., 'fit'
        or 'predict').

    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    config: StageConfig
    input_validation: dict[str, Data] = Field(default_factory=dict)
    output_validation: dict[str, Data] = Field(default_factory=dict)
    _dataif: DataInterface | None = None
    _module: Path | None = None
    _input: Input | None = None
    _required_input: set[str] = set()
    _optional_input: set[str] = set()
    _output: set[str] = set()
    _skip: set[str] = set()

    @property
    def dataif(self) -> DataInterface:
        """Stage data interface.

        Examples
        --------
        Load input file:
        * _requred_input: {"data.parquet"}
        * data = self.dataif.load(key="data")

        Load file from input directory:
        * _required_input: {"submodels"}
        * model = self.dataif.load(f"model_{subset_id}.pkl", key="submodels")

        Load output file:
        * subsets = self.dataif.load("subsets.csv", key="output")

        Dump output file:
        * self.dataif.dump(f"submodels/model_{subset_id}.pkl", key="output")

        """
        if self._dataif is None:
            raise AttributeError(f"Stage '{self.name}' dataif has not been set")
        return self._dataif

    def set_dataif(self, config_path: Path | str) -> None:
        """Set stage data interface.

        Parameters
        ----------
        config_path : Path or str
            Path to config file.

        Notes
        -----
        * This method is called in Pipeline.build.
        * This method assumes the pipeline's data flow has already been
          defined (i.e., if the stage's input is changed after pipeline
          is built, the data interface will not contain the new input).

        """
        directory = Path(config_path).parent
        self._dataif = DataInterface(
            directory=directory,
            config=config_path,
            output=directory / self.name,
        )
        for item_name, item_value in self.input.items.items():
            if isinstance(item_value, Path):
                self._dataif.add_path(item_name, item_value)
            elif isinstance(item_value, Data):
                item_value.path = directory / item_value.stage / item_value.path
                self._dataif.add_path(item_name, item_value.path)
        for item_name, item_value in self.output.items.items():
            item_value.path = directory / self.name / item_value.path
        if not (directory / self.name).exists():
            (directory / self.name).mkdir()

    @computed_property
    def module(self) -> Path | None:
        if self._module is None and not hasattr(
            onemod_stages, self.type
        ):  # custom stage
            try:
                return Path(getfile(self.__class__))
            except TypeError:
                raise TypeError(f"Could not find module for {self.name} stage")
        return self._module

    @property
    def skip(self) -> set[str]:
        return self._skip

    @computed_property
    def input(self) -> Input | None:
        if self._input is None:
            self._input = Input(
                stage=self.name,
                required=self._required_input,
                optional=self._optional_input,
            )
        return self._input

    @cached_property
    def output(self) -> Output:
        output_items = {}
        for item in self._output:
            item_specs = item.split(".")
            item_name = item_specs[0]
            item_type = "directory" if len(item_specs) == 1 else item_specs[1]
            output_items[item_name] = Data(
                stage=self.name, path=Path(item), format=item_type
            )
        return Output(stage=self.name, items=output_items)

    @property
    def dependencies(self) -> set[str]:
        if self.input is None:
            return set()
        return self.input.dependencies

    @computed_property
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
        with open(config_path, "r") as file:
            pipeline_config = json.load(file)
        try:
            stage_config = pipeline_config["stages"][stage_name]
        except KeyError:
            raise AttributeError(
                f"{pipeline_config['name']} does not contain a stage named '{stage_name}'"
            )
        stage = cls(**stage_config)
        stage.config.inherit(pipeline_config["config"])
        if "module" in stage_config:
            stage._module = stage_config["module"]
        if hasattr(stage, "apply_stage_specific_config"):
            stage.apply_stage_specific_config(stage_config)
        if (input := stage_config.get("input")) is not None:
            stage(**input)
        stage.set_dataif(config_path)
        return stage

    @validate_call
    def evaluate(
        self,
        method: Literal["run", "fit", "predict", "collect"] = "run",
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
        resources : Path, str, or dict, optional
            Dictionary of compute resources or path to resources file.
            Required if `backend` is 'jobmon'.

        Notes
        -----
        If stage does not implement `method`, and `method` not in
        `skip`, the `run` method will be evaluated. For example, a data
        preprocessing stage might implement `run`, skip `predict`, and
        evaluate `run` when `fit` is called. Alternatively, a plotting
        stage might implement `run`, skip `fit`, and evaluate `run` when
        `predict` is called.

        """
        if method == "collect":
            raise ValueError(
                "Method 'collect' can only be called for 'ModelStage' objects"
            )

        if method in self.skip:
            warnings.warn(f"'{self.name}' stage skips the '{method}' method")
            return

        method = method if hasattr(self, method) else "run"

        self.input.check_exists()

        if backend == "jobmon":
            from onemod.backend.jobmon_backend import evaluate_with_jobmon

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
                    schema.validate_data(None, collector)
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
                    data_spec.validate_data(None, collector)
                else:
                    handle_error(
                        self.name,
                        "Output Validation",
                        KeyError,
                        f"Output data '{item_name}' not found after stage execution.",
                        collector,
                    )

    def get_field(self, field: str, stage_name: str | None = None) -> Any:
        """Get field from config file.

        Parameters
        ----------
        field : str
            Name of field. If field is nested, join keys with '-'.
            For example, 'config-id_columns`.
        stage_name : str or None, optional
            Name of stage if field belongs to stage. Default is None.

        Returns
        -------
        Any
            Field item.

        """
        config = self.dataif.load(key="config")
        if stage_name is not None:
            config = config["stages"][stage_name]
        for key in field.split("-"):
            config = config[key]
        return config

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """Run stage."""
        raise NotImplementedError("Subclasses must implement this method.")

    @validate_call
    def __call__(self, **input: Data | Path) -> Output:
        """Define stage dependencies."""
        # FIXME: Update data interface if it exists?
        self.input.check_missing({**self.input.items, **input})
        self.input.update(input)
        return self.output

    def __repr__(self) -> str:
        return f"{self.type}(name={self.name})"


class ModelStage(Stage, ABC):
    """Model stage base class.

    Model stages can be run separately for data subsets using the
    `groupby` attribute. For example, a single stage could have separate
    models by sex_id or age_group_id.

    Model stages can also be run for different parameter combinations
    using the `crossby` attribute. For example, a single stage could be
    run for various hyperparameter values, and then the results could be
    combined into an ensemble. Any parameter in
    `config.crossable_params` can be specified as either a single value
    or a list of values.

    When a model stage method is evaluated, all submodels (identified by
    their `subset_id` and `param_id`) are evaluated, and then, if
    `method` is in `collect_after`, the submoel results are collected
    using the `collect` method.

    Parameters
    ----------
    name : str
        Stage name.
    config : StageConfig
        Stage configuration.
    groupby : set of str or None, optional
        Column names used to create data subsets.
        Default is None.
    input_validation : dict, optional
        Description.
    output_validation : dict, optional
        Description.

    Notes
    -----
    * Private attributes that are defined automatically:
      * `_dataif : `DataInterface` object for loading/dumping input,
        created `Stage.set_dataif()` and called by `Pipeline.build()`
        or `Stage.from_json()`.
      * `_module`: Path to custom stage definition, created in
        `Stage.module` or `Stage.from_json()`.
      * `_input`: `Input` object that organizes `Stage` input, created
        in `Stage.input` or `Stage.from_json()`, modified by
        `Stage.__call__()`.
      * `_crossby`: Names of parameters using multiple values. Created
        in `ModelStage.create_stage_params()` AND CALLED BY
      * `_subset_ids`: Data subset ID values. Created in
        `ModelStage.create_stage_subsets()` and CALLED BY
      * `_param_ids`: Parameter set ID values. Created in
        `ModelStage.create_stage_params()` and CALLED BY
    * Private attributes that must be defined by class:
      * `_required_input`, `_optional_input`, `_output`: Strings with
        syntax "f{name}.{extension}". For example, "data.parquet"
        (required to use `groupby` attribute). If input/output is a
        directory instead of a file, exclude the extension. For example,
        "submodels".
      * `_skip`: Methods that the stage does not implement (e.g., 'fit'
        or 'predict').
      * `_collect_after`: Methods that create submodel results (e.g.,
        data subsets or parameter sets) that must be collected. For
        example, collect submodel results for parameter sets to select
        best parameter values after the 'fit'  method, or collect
        submodel results for data subsets after the 'predict' method.

    """

    config: StageConfig
    groupby: set[str] | None = None
    _crossby: set[str] | None = None
    _subset_ids: set[int] = set()
    _param_ids: set[int] = set()
    _collect_after: set[str] = set()

    @computed_property
    def crossby(self) -> set[str] | None:
        return self._crossby

    @property
    def subset_ids(self) -> set[int]:
        if self.groupby is not None and not self._subset_ids:
            try:
                subsets = self.dataif.load("subsets.csv", key="output")
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
                params = self.dataif.load("parameters.csv", key="output")
                self._param_ids = set(params["param_id"])
            except FileNotFoundError:
                raise AttributeError(
                    f"{self.name} parameter sets have not been created"
                )
        return self._param_ids

    @property
    def collect_after(self) -> set[str]:
        return self._collect_after

    def apply_stage_specific_config(self, stage_config: dict) -> None:
        """Apply ModelStage-specific configuration."""
        if "crossby" in stage_config:
            self._crossby = stage_config["crossby"]

    def create_stage_subsets(
        self, data_key: str, id_subsets: dict[str, list[Any]] | None = None
    ) -> None:
        """Create stage data subsets from groupby and id_subsets."""
        if self.groupby is None:
            raise AttributeError(
                f"{self.name} does not have a groupby attribute"
            )

        df = self.dataif.load(
            key=data_key,
            columns=list(self.groupby),
            id_subsets=id_subsets,
            return_type="pandas_dataframe",
        )

        subsets_df = create_subsets(self.groupby, df)
        self._subset_ids = set(subsets_df["subset_id"].to_list())

        self.dataif.dump(subsets_df, "subsets.csv", key="output")

    def get_stage_subset(
        self, subset_id: int, *fparts: str, key: str = "data", **options
    ) -> DataFrame:
        """Filter data by stage subset_id."""
        return get_subset(
            self.dataif.load(*fparts, key=key, **options),
            self.dataif.load("subsets.csv", key="output"),
            subset_id,
        )

    def create_stage_params(self) -> None:
        """Create stage parameter sets from config."""
        params = create_params(self.config)
        if params is not None:
            if "param_id" not in params.columns:
                raise KeyError("Parameter set ID column 'param_id' not found")

            self._crossby = set(params.columns) - {"param_id"}
            self._param_ids = set(params["param_id"])
            self.dataif.dump(params, "parameters.csv", key="output")

    def set_params(self, param_id: int) -> None:
        """Set stage parameters."""
        params = get_params(
            self.dataif.load("parameters.csv", key="output"), param_id
        )
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
            Submodel data subset ID. Ignored if `backend` is 'jobmon' or
            method is `collect`.
        param_id : int, optional
            Submodel parameter set ID. Ignored if `backend` is 'jobmon'
            or method is `collect`.
        cluster : str, optional
            Cluster name. Required if `backend` is 'jobmon'.
        resources : Path, str, or dict, optional
            Dictionary of compute resources or path to resources file.
            Required if `backend` is 'jobmon'.

        Notes
        -----
        If either `subset_id` or `param_id` is passed, `method` will be
        evaluated for the corresponding submodel (unless `backend` is
        'jobmon' or `method` is 'collect', in which case `subset_id`
        and `param_id` are ignored). Otherwise, `method` will be
        evaluated for all submodels, and then, if `method` is in
        `collect_after`, the `collect` method will be evaluated to
        collect the submodel results.

        """
        if method in self.skip:
            warnings.warn(f"{self.name} skips the '{method}' method")
            return

        self.input.check_exists()

        if backend == "jobmon":
            if method == "collect":
                raise ValueError(
                    "Method 'collect' cannot be used with 'jobmon' backend"
                )

            from onemod.backend.jobmon_backend import evaluate_with_jobmon

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
                    from onemod.backend.local_backend import evaluate_local

                    evaluate_local(model=self, method=method)

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """Run stage submodel."""
        raise NotImplementedError("Subclasses must implement this method.")

    def fit(self, *args, **kwargs) -> None:
        """Fit stage submodel."""
        raise NotImplementedError(
            "Subclasses must implement this method if not skipped."
        )

    def predict(self, *args, **kwargs) -> None:
        """Predict stage submodel."""
        raise NotImplementedError(
            "Subclasses must implement this method if not skipped."
        )

    @abstractmethod
    def collect(self) -> None:
        """Collect stage submodel results."""
        raise NotImplementedError("Subclasses must implement this method.")

    def __repr__(self) -> str:
        stage_str = f"{self.type}(name={self.name}"
        if self.groupby is not None:
            stage_str += f", groupby={self.groupby}"
        if self.crossby is not None:
            stage_str += f", crossby={self.crossby}"
        return stage_str + ")"
