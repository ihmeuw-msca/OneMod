"""Stage base classes."""

from __future__ import annotations

import json
from abc import ABC
from functools import cached_property
from inspect import getfile
from itertools import product
from pathlib import Path
from typing import Any, Literal, Mapping

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, validate_call

import onemod.stage as onemod_stages
from onemod.config import StageConfig
from onemod.dtypes import Data
from onemod.dtypes.unique_sequence import UniqueList
from onemod.fsutils import DataInterface
from onemod.io import Input, Output
from onemod.utils.decorators import computed_property
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
    submodels are evaluated, and then, if `method` is in
    `collect_after`, the submodel results are collected using the
    `collect` method.

    Parameters
    ----------
    name : str
        Stage name.
    config : StageConfig, optional
        Stage configuration.
    groupby : list of str or None, optional
        Column names used to create submodel data subsets. Default is
        None.
    crossby : list of str or None, optional
        Parameter names used to create submodel parameter sets. Default
        is None.
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
      * `_subsets`: Data subsets. Created in
        `Stage.create_stage_subsets().
      * `_paramsets`: Parameter sets. Created in
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
    config: StageConfig = StageConfig()
    groupby: UniqueList[str] | None = None
    crossby: UniqueList[str] | None = None
    input_validation: dict[str, Data] = {}
    output_validation: dict[str, Data] = {}
    _dataif: DataInterface | None = None
    _module: Path | None = None
    _input: Input | None = None
    _required_input: list[str] = []
    _optional_input: list[str] = []
    _output: list[str] = []
    _skip: list[str] = []
    _collect_after: list[str] = []
    _subsets: DataFrame | None = None
    _paramsets: DataFrame | None = None

    @property
    def dataif(self) -> DataInterface:
        """Stage data interface.

        Examples
        --------
        Load input file:
        * _requred_input: ["data.parquet"]
        * data = self.dataif.load(key="data")

        Load file from input directory:
        * _required_input: ["submodels"]
        * model = self.dataif.load("model_0.pkl", key="submodels")

        Load output file:
        * subsets = self.dataif.load("subsets.csv", key="output")

        Dump output file:
        * self.dataif.dump("submodels/model_0.pkl", key="output")

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
    def skip(self) -> list[str]:
        return self._skip

    @property
    def collect_after(self) -> list[str]:
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
    def dependencies(self) -> list[str]:
        if self.input is None:
            return []
        return self.input.dependencies

    @computed_property
    def type(self) -> str:
        return type(self).__name__

    @property
    def has_submodels(self) -> bool:
        return self.groupby is not None or self.crossby is not None

    def get_submodels(
        self,
        subsets: dict[str, Any | list[Any]] | None = None,
        paramsets: dict[str, Any | list[Any]] | None = None,
    ) -> list[tuple[dict[str, Any] | None, ...]]:
        """Get stage data subset/parameter set combinations.

        Parameters
        ----------
        subsets : dict or None, optional
            Submodel data subsets to include. If None, include all
            submodel data subsets. Default is None.
        paramsets : dict or None, optional
            Submodel parameter sets to include. If None, include all
            submodel parameter sets. Default is None.

        Returns
        -------
        list of tuple
            Submodel data subset/parameter set combinations.

        """
        if not self.has_submodels:
            raise AttributeError(f"Stage '{self.name}' does not have submodels")
        if subsets is not None and self.subsets is None:
            raise AttributeError(
                f"Stage '{self.name}' does not use groupby attribute"
            )
        if paramsets is not None and self.paramsets is None:
            raise AttributeError(
                f"Stage '{self.name}' does not use crossby attribute"
            )

        # Filter data subsets and parameter sets
        filtered_subsets = (
            self.subsets
            if subsets is None
            else self.get_subset(self.subsets, subsets)
        )
        filtered_paramsets = (
            self.paramsets
            if paramsets is None
            else self.get_subset(self.paramsets, paramsets)
        )

        # Generate all data subset/parameter set combinations
        return list(
            product(
                [None]
                if filtered_subsets is None
                else filtered_subsets.to_dict(orient="records"),
                [None]
                if filtered_paramsets is None
                else filtered_paramsets.to_dict(orient="records"),
            )
        )

    @property
    def subsets(self) -> DataFrame | None:
        if self.groupby is not None and self._subsets is None:
            try:
                self._subsets = self.dataif.load("subsets.csv", key="output")
            except FileNotFoundError:
                raise AttributeError(
                    f"Stage '{self.name}' submodel data subsets have not been created"
                )
        return self._subsets

    @property
    def paramsets(self) -> DataFrame | None:
        if self.crossby is not None and self._paramsets is None:
            try:
                self._paramsets = self.dataif.load(
                    "paramsets.csv", key="output"
                )
            except FileNotFoundError:
                raise AttributeError(
                    f"Stage '{self.name}' submodel parameter sets have not been created"
                )
        return self._paramsets

    def create_subsets(self, groupby_data: Path | str) -> None:
        """Create submodel data subsets from groupby."""
        if self.groupby is None:
            raise AttributeError(
                f"Stage '{self.name}' does not use groupby attribute"
            )

        data = self.dataif.load(
            str(groupby_data), key="", columns=self.groupby
        ).drop_duplicates()
        groups = data.groupby(self.groupby)
        self._subsets = DataFrame(
            [subset for subset in groups.groups.keys()], columns=self.groupby
        ).sort_values(by=self.groupby)
        self.dataif.dump(self._subsets, "subsets.csv", key="output")

    @staticmethod
    def get_subset(
        data: DataFrame, subset: Mapping[str, Any | list[Any]]
    ) -> DataFrame:
        """Filter data by subset."""
        for col, values in subset.items():
            data = data[
                data[col].isin(values if isinstance(values, list) else [values])
            ]

        if len(data) == 0:
            raise ValueError(f"Empty subset or paramset: {subset}")

        return data.reset_index(drop=True)

    def create_params(self) -> None:
        """Create submodel parameter sets from crossby."""
        if self.crossby is None:
            raise AttributeError(
                f"Stage '{self.name}' does not use crossby attribute"
            )

        # Get all parameters with multiples values from config
        param_dict = {}
        for param_name in self.crossby:
            param_values = self.config[param_name]
            if isinstance(param_values, (list, set, tuple)):
                param_dict[param_name] = param_values
            else:
                raise ValueError(
                    f"Crossby param '{param_name}' must be a list, set, or tuple"
                )

        # Create parameter sets
        self._paramsets = DataFrame(
            list(product(*param_dict.values())), columns=list(param_dict.keys())
        ).sort_values(by=self.crossby)
        self.dataif.dump(self._paramsets, "paramsets.csv", key="output")

    def set_params(self, paramset: dict[str, Any]) -> None:
        """Set submodel parameters."""
        if self.crossby is None:
            raise AttributeError(
                f"Stage '{self.name}' does not use crossby attribute"
            )

        for param_name in self.crossby:
            if param_name not in paramset:
                raise KeyError(
                    f"Stage '{self.name}' param set missing param: {param_name}"
                )
            self.config[param_name] = paramset[param_name]

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

    def evaluate(
        self,
        method: Literal["run", "fit", "predict", "collect"],
        backend: Literal["local", "jobmon"],
        **kwargs,
    ) -> None:
        """Evaluate stage method."""
        if backend == "jobmon":
            from onemod.backend.jobmon_backend import evaluate
        else:
            from onemod.backend.local_backend import evaluate  # type: ignore

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

    def get_field(
        self, field: str, stage_name: str | None = None, default: Any = None
    ) -> Any:
        """Get field from config file.

        Parameters
        ----------
        field : str
            Name of field. If field is nested, join keys with ':'.
            For example, 'config:param`.
        stage_name : str or None, optional
            Name of stage if field belongs to stage. Default is None.

        Returns
        -------
        Any
            Field item.

        """
        config = self.dataif.load(key="config")
        if stage_name is not None:
            config = config.get("stages", {}).get(stage_name, {})
        for key in field.split(":"):
            config = config.get(key, {})
        return config or default

    @validate_call
    def run(
        self,
        backend: Literal["local", "jobmon"] = "local",
        subsets: dict[str, Any | list[Any]] | None = None,
        paramsets: dict[str, Any | list[Any]] | None = None,
        collect: bool = False,
        **kwargs,
    ) -> None:
        """Run stage.

        Parameters
        ----------
        backend : {'local', 'jobmon'}, optional
            How to evaluate the method. Default is 'local'.
        subsets : dict, optional
            Submodel data subsets to evaluate. If None, evaluate all
            data subsets. Default is None.
        paramsets : dict, optional
            Submodel parameter sets to evaluate. If None, evaluate all
            data subsets. Default is None.
        collect : bool, optional
            Collect submodel results if `subsets` and `paramsets` are
            not both None. Default is False. If `subsets` and
            `paramsets` are not both None, this parameter is ignored and
            submodel results are always collected.

        Other Parameters
        ----------------
        method_args : dict, optional
            Additional keyword arguments passed to stage methods.
        cluster : str, optional
            Cluster name. Required if `backend` is 'jobmon'.
        resources : Path, str, or dict, optional
            Path to resources file or dictionary of compute resources.
            Required if `backend` is 'jobmon'.

        """
        self.evaluate(
            "run",
            backend,
            subsets=subsets,
            paramsets=paramsets,
            collect=collect,
            **kwargs,
        )

    def _run(self, *args, **kwargs) -> None:
        """Run stage submodel.

        Parameters
        ----------
        subset : dict[str, Any]
            If stage uses `groupby` attribute, add `subset` arg when
            implementing `_run`.
        paramset : dict[str, Any]
            If stage uses `crossby` attribute, add `paramset` arg when
            implementing `_run`.
        **kwargs
            Additional keyword arguments can be passed directly to
            `_run` method, but must be passed via `method_args`
            dictionary to command line or `run` method.

        """
        raise NotImplementedError("Subclasses must implement this method")

    @validate_call
    def fit(
        self,
        backend: Literal["local", "jobmon"] = "local",
        subsets: dict[str, Any | list[Any]] | None = None,
        paramsets: dict[str, Any | list[Any]] | None = None,
        collect: bool = False,
        **kwargs,
    ) -> None:
        """Fit stage.

        Parameters
        ----------
        backend : {'local', 'jobmon'}, optional
            How to evaluate the method. Default is 'local'.
        subsets : dict, optional
            Submodel data subsets to evaluate. If None, evaluate all
            data subsets. Default is None.
        paramsets : dict, optional
            Submodel parameter sets to evaluate. If None, evaluate all
            data subsets. Default is None.
        collect : bool, optional
            Collect submodel results if `subsets` and `paramsets` are
            not both None. Default is False. If `subsets` and
            `paramsets` are not both None, this parameter is ignored and
            submodel results are always collected.

        Other Parameters
        ----------------
        method_args : dict, optional
            Additional keyword arguments passed to stage methods.
        cluster : str, optional
            Cluster name. Required if `backend` is 'jobmon'.
        resources : Path, str, or dict, optional
            Path to resources file or dictionary of compute resources.
            Required if `backend` is 'jobmon'.

        """
        self.evaluate(
            "fit",
            backend,
            subsets=subsets,
            paramsets=paramsets,
            collect=collect,
            **kwargs,
        )

    def _fit(self, *args, **kwargs) -> None:
        """Fit stage submodel.

        Parameters
        ----------
        subset : dict[str, Any]
            If stage uses `groupby` attribute, add `subset` arg when
            implementing `_fit`.
        paramset : dict[str, Any]
            If stage uses `crossby` attribute, add `paramset` arg when
            implementing `_fit`.
        **kwargs
            Additional keyword arguments can be passed directly to
            `_fit` method, but must be passed via `method_args`
            dictionary to command line or `fit` method.

        """
        raise NotImplementedError(
            "Subclasses must implement this method if not skipped"
        )

    @validate_call
    def predict(
        self,
        backend: Literal["local", "jobmon"] = "local",
        subsets: dict[str, Any | list[Any]] | None = None,
        paramsets: dict[str, Any | list[Any]] | None = None,
        collect: bool = False,
        **kwargs,
    ) -> None:
        """Create stage predictions.

        Parameters
        ----------
        backend : {'local', 'jobmon'}, optional
            How to evaluate the method. Default is 'local'.
        subsets : dict, optional
            Submodel data subsets to evaluate. If None, evaluate all
            data subsets. Default is None.
        paramsets : dict, optional
            Submodel parameter sets to evaluate. If None, evaluate all
            data subsets. Default is None.
        collect : bool, optional
            Collect submodel results if `subsets` and `paramsets` are
            not both None. Default is False. If `subsets` and
            `paramsets` are not both None, this parameter is ignored and
            submodel results are always collected.

        Other Parameters
        ----------------
        cluster : str, optional
            Cluster name. Required if `backend` is 'jobmon'.
        resources : Path, str, or dict, optional
            Path to resources file or dictionary of compute resources.
            Required if `backend` is 'jobmon'.
        method_args : dict, optional
            Additional keyword arguments passed to stage methods.

        """
        self.evaluate(
            "fit",
            backend,
            subsets=subsets,
            paramsets=paramsets,
            collect=collect,
            **kwargs,
        )

    def _predict(self, *args, **kwargs) -> None:
        """Create stage submodel predictions.

        Parameters
        ----------
        subset : dict[str, Any]
            If stage uses `groupby` attribute, add `subset` arg when
            implementing `_predict`.
        paramset : dict[str, Any]
            If stage uses `crossby` attribute, add `paramset` arg when
            implementing `_predict`.
        **kwargs
            Additional keyword arguments can be passed directly to
            `_predict` method, but must be passed via `method_args`
            dictionary to command line or `predict` method.

        """
        raise NotImplementedError(
            "Subclasses must implement this method if not skipped"
        )

    @validate_call
    def collect(self) -> None:
        """Collect stage submodel results."""
        raise NotImplementedError(
            "Subclasses must implement this method if using submodels"
            " and collect_after not empty"
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
        if self.groupby is not None:
            stage_str += f", groupby={self.groupby}"
        if self.crossby is not None:
            stage_str += f", crossby={self.crossby}"
        return stage_str + ")"
