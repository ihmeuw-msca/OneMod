"""Stage base classes."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import cached_property
from inspect import getfile
from pathlib import Path
from typing import Any, Generator, Literal

from pandas import DataFrame
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    validate_call,
)

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

    Stages can be run separately for data subsets using the `groupby`
    attribute. For example, a single stage can have separate models for
    each sex_id - age_group_id pair.

    Stages can also be run for different parameter combinations using
    the `crossby` attribute. For example, a single stage can be run for
    various hyperparameter values, and then the results can be combined
    into an ensemble.

    When a stage using `groupby` and/or `crossby` is evaluated, all
    submodels (identified by their `subset_id` and `param_id`) are
    evaluated, and then, if `method` is in `collect_after`, the submodel
    results are collected using the `collect` method.

    Parameters
    ----------
    name : str
        Stage name.
    config : StageConfig
        Stage configuration.
    groupby : tuple of str, optional
        Column names used to create data subsets. Default is an empty
        tuple.
    crossby : tuple of str, optional
        Parameter names used to create parameter sets. Default is an
        empty tuple.
    input_validation : dict, optional
        Optional specification of input data validation.
    output_validation : dict, optional
        Optional specification of output data validation.

    Notes
    -----
    * Private attributes that are defined automatically:
      * `_dataif : `DataInterface` object for loading/dumping input,
        created in `Stage.set_dataif()` and called by `Pipeline.build()`
        or `Stage.from_json()`.
      * `_module`: Path to custom stage definition, created in
        `Stage.module` or `Stage.from_json()`.
      * `_input`: `Input` object that organizes `Stage` input, created
        in `Stage.input` or `Stage.from_json()`, modified by
        `Stage.__call__()`.
      * `_subset_ids`: Data subset ID values. Created in
        `Stage.create_stage_subsets().
      * `_param_ids`: Parameter set ID values. Created in
        `Stage.create_stage_params()`.
    * Private attributes that must be defined by class:
      * `_required_input`, `_optional_input`, `_output`: Strings with
        syntax "f{name}.{extension}". For example, "data.parquet". If
        input/output is a directory instead of a file, exclude the
        extension. For example, "submodels".
      * `_skip`: Methods that the stage does not implement (e.g., 'fit'
        or 'predict').
      * `_collect_after`: Methods that create submodel results (e.g.,
        data subsets or parameter sets) that must be collected. For
        example, collect submodel results for parameter sets to select
        best parameter values after the 'fit' method, or collect
        submodel results for data subsets after the 'predict' method. Do
        not put 'collect' in `_collect_after`!

    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    config: StageConfig
    groupby: tuple[str, ...] = tuple()
    crossby: tuple[str, ...] = tuple()
    input_validation: dict[str, Data] = Field(default_factory=dict)
    output_validation: dict[str, Data] = Field(default_factory=dict)
    _dataif: DataInterface | None = None
    _module: Path | None = None
    _input: Input | None = None
    _required_input: set[str] = set()
    _optional_input: set[str] = set()
    _output: set[str] = set()
    _skip: set[str] = set()
    _collect_after: set[str] = set()
    _subset_ids: tuple[int, ...] = tuple()
    _param_ids: tuple[int, ...] = tuple()

    @field_validator("groupby", "crossby", mode="after")
    @classmethod
    def unique_tuple(cls, items: tuple[str, ...]) -> tuple[str, ...]:
        """Make sure groupby and crossby have unique values."""
        if len(items) > 0:
            return tuple(dict.fromkeys(items))
        return items

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

    @property
    def collect_after(self) -> set[str]:
        return self._collect_after

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

    @property
    def has_submodels(self) -> bool:
        return len(self.groupby) > 0 or len(self.crossby) > 0

    @property
    def submodel_ids(self) -> Generator:
        if not self.has_submodels:
            raise AttributeError(f"Stage '{self.name}' does not have submodels")

        if len(self.subset_ids) > 0:
            for subset_id in self.subset_ids:
                if len(self.param_ids) > 0:
                    for param_id in self.param_ids:
                        yield (subset_id, param_id)
                else:
                    yield (subset_id, None)
        else:
            for param_id in self.param_ids:
                yield (None, param_id)

    @property
    def subset_ids(self) -> tuple[int, ...]:
        if len(self.groupby) > 0 and len(self._subset_ids) == 0:
            try:
                subsets = self.dataif.load("subsets.csv", key="output")
                self._subset_ids = tuple(subsets["subset_id"])
            except FileNotFoundError:
                raise AttributeError(
                    f"Stage '{self.name}' data subsets have not been created"
                )
        return self._subset_ids

    @property
    def param_ids(self) -> tuple[int, ...]:
        if len(self.crossby) > 0 and len(self._param_ids) == 0:
            try:
                params = self.dataif.load("parameters.csv", key="output")
                self._param_ids = tuple(params["param_id"])
            except FileNotFoundError:
                raise AttributeError(
                    f"Stage '{self.name}' parameter sets have not been created"
                )
        return self._param_ids

    def create_stage_subsets(
        self, data_key: str, id_subsets: dict[str, list[Any]] | None = None
    ) -> None:
        """Create stage data subsets from groupby and id_subsets."""
        if len(self.groupby) == 0:
            raise AttributeError(
                f"Stage '{self.name}' does not use groupby attribute"
            )

        df = self.dataif.load(
            key=data_key, columns=list(self.groupby), id_subsets=id_subsets
        )

        subsets_df = create_subsets(self.groupby, df)
        self._subset_ids = tuple(subsets_df["subset_id"])
        self.dataif.dump(subsets_df, "subsets.csv", key="output")

    def get_stage_subset(
        self, subset_id: int, *fparts: str, key: str = "data", **options
    ) -> DataFrame:
        """Filter data by stage subset_id."""
        if len(self.groupby) == 0:
            raise AttributeError(
                f"Stage '{self.name}' does not use groupby attribute"
            )

        return get_subset(
            self.dataif.load(*fparts, key=key, **options),
            self.dataif.load("subsets.csv", key="output"),
            subset_id,
        )

    def create_stage_params(self) -> None:
        """Create stage parameter sets from crossby and config."""
        if len(self.crossby) == 0:
            raise AttributeError(
                f"Stage '{self.name}' does not use crossby attribute"
            )

        params_df = create_params(self.crossby, self.config)
        self._param_ids = tuple(params_df["param_id"])
        self.dataif.dump(params_df, "parameters.csv", key="output")

    def set_params(self, param_id: int) -> None:
        """Set stage parameters."""
        if len(self.crossby) == 0:
            raise AttributeError(
                f"Stage '{self.name}' does not use crossby attribute"
            )

        params = get_params(
            self.dataif.load("parameters.csv", key="output"), param_id
        )

        for param_name, param_value in params.items():
            self.config[param_name] = param_value

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
        stage.config.add_pipeline_config(pipeline_config["config"])
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
        subset_id : int, optional
            Submodel data subset ID. Ignored if `backend` is 'jobmon' or
            `method` is 'collect'.
        param_id : int, optional
            Submodel parameter set ID. Ignored if `backend` is 'jobmon'
            or `method` is 'collect'.
        cluster : str, optional
            Cluster name. Required if `backend` is 'jobmon'.
        resources : Path, str, or dict, optional
            Dictionary of compute resources or path to resources file.
            Required if `backend` is 'jobmon'.

        Notes
        -----
        If `subset_id` and/or `param_id` are passed, `method` will be
        evaluated for the corresponding submodel. Otherwise, `method`
        will be evaluated for all submodels, and then, if `method` is in
        `collect_after`, the `collect` method will be evaluated to
        collect the submodel results.

        """
        if backend == "jobmon":
            from onemod.backend.jobmon_backend import evaluate
        else:
            from onemod.backend.local_backend import evaluate

        evaluate(model=self, method=method, **kwargs)

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

    def fit(self, *args, **kwargs) -> None:
        """Fit stage."""
        raise NotImplementedError(
            "Subclasses must implement this method if not skipped."
        )

    def predict(self, *args, **kwargs) -> None:
        """Predict stage submodel."""
        raise NotImplementedError(
            "Subclasses must implemented this method if not skipped."
        )

    def collect(self) -> None:
        """Collect stage submodel results."""
        raise NotImplementedError(
            "Subclasses must implement this method if collect_after not empty."
        )

    @validate_call
    def __call__(self, **input: Data | Path) -> Output:
        """Define stage dependencies."""
        # FIXME: Update data interface if it exists?
        self.input.check_missing({**self.input.items, **input})
        self.input.update(input)
        return self.output

    def __repr__(self) -> str:
        stage_str = f"{self.type}(name={self.name}"
        if len(self.groupby) > 0:
            stage_str += f", groupby={self.groupby}"
        if len(self.crossby) > 0:
            stage_str += f", crossby={self.crossby}"
        return stage_str + ")"
