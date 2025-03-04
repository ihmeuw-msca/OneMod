"""Stage base classes."""

from __future__ import annotations

import json
from abc import ABC
from inspect import getfile
from itertools import product
from pathlib import Path
from typing import Any, Literal, Mapping

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict

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

    Parameters
    ----------
    name : str
        Stage name.
    config : StageConfig, optional
        Stage configuration.
    groupby : list of str, optional
        Column names used to create submodel data subsets.
    crossby : list of str, optional
        Parameter names used to create submodel parameter sets.
    input_validation : dict of str: Data, optional
        Optional specification of input data validation.
    output_validation : dict of str: Data, optional
        Optional specification of output data validation.

    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    config: StageConfig = StageConfig()
    groupby: UniqueList[str] | None = None
    crossby: UniqueList[str] | None = None
    input_validation: dict[str, Data] | None = None
    output_validation: dict[str, Data] | None = None
    _module: Path | None = None
    _input: Input
    _output: Output
    _dataif: DataInterface
    _subsets: DataFrame | None = None
    _paramsets: DataFrame | None = None
    _skip: list[str] = []
    _collect_after: list[str] = []
    _required_input: dict[str, dict[str, Any]] = {}
    _optional_input: dict[str, dict[str, Any]] = {}
    _output_items: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        module: Path | str | None = None,
        input: Input | dict = {},
        config_path: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Create stage instance."""
        super().__init__(**kwargs)
        self.set_module(module)
        self.set_input(input)
        if config_path is not None:
            self.set_dataif(config_path)
            self.set_output()

    @computed_property
    def type(self) -> str:
        """Stage type."""
        return type(self).__name__

    @computed_property
    def module(self) -> Path | None:
        """Path to module containing custom stage definition."""
        return self._module

    def set_module(self, module: Path | str | None) -> None:
        if isinstance(module, (Path, str)):
            self._module = Path(module)
        else:
            if not hasattr(onemod_stages, self.type):
                try:
                    self._module = Path(getfile(self.__class__))
                except (OSError, TypeError):
                    raise TypeError(
                        f"Could not find module for custom stage '{self.name}'"
                    )

    @computed_property
    def input(self) -> Input:
        """Stage input metadata."""
        # TODO: Could add more description, like keys, Data
        return self._input

    def set_input(self, input: Data | dict) -> None:
        self._input = Input(
            stage=self.name,
            required=self._required_input,
            optional=self._optional_input,
            items=input,
        )

    @property
    def output(self) -> Output:
        """Stage output metadata."""
        # TODO: Could add more description, like keys, Data
        return self._output

    def set_output(self) -> None:
        self._output = Output(
            stage=self.name,
            directory=self.dataif.get_path("output"),
            items=self._output_items,
        )

    @property
    def dependencies(self) -> list[str]:
        """Stage dependencies."""
        return self.input.dependencies

    @property
    def dataif(self) -> DataInterface:
        """Stage data interface."""
        # TODO: Add more detailed description and examples
        return self._dataif

    def set_dataif(self, config_path: Path | str) -> None:
        # Create data interface
        directory = Path(config_path).parent
        self._dataif = DataInterface(
            config=config_path,
            directory=directory,
            output=directory / self.name,
        )

        # Add input items
        for item_name, item_value in self.input.items.items():
            if item_value.path is not None:
                self._dataif.add_path(item_name, item_value.path)

    @property
    def skip(self) -> list[str]:
        """Names of methods skipped by the stage."""
        return self._skip

    @property
    def subsets(self) -> DataFrame | None:
        """Stage data subsets."""
        if self.groupby is not None and self._subsets is None:
            try:
                self._subsets = self.dataif.load("subsets.csv", key="output")
            except FileNotFoundError:
                raise AttributeError(
                    f"Stage '{self.name}' submodel data subsets have not been created"
                )
        return self._subsets

    def create_subsets(self, groupby_data: Path | str) -> None:
        """Create submodel data subsets from groupby."""
        if self.groupby is None:
            raise AttributeError(
                f"Stage '{self.name}' does not use groupby attribute"
            )

        data = self.dataif.load(
            str(groupby_data), columns=self.groupby
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

    @property
    def paramsets(self) -> DataFrame | None:
        """Stage parameter sets."""
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

    @property
    def has_submodels(self) -> bool:
        """Whether the stage has submodels."""
        return self.groupby is not None or self.crossby is not None

    def get_submodels(
        self,
        subsets: dict[str, Any | list[Any]] | None = None,
        paramsets: dict[str, Any | list[Any]] | None = None,
    ) -> list[tuple[dict[str, Any] | None, ...]]:
        """Get stage submodels.

        A stage submodel consists of a single ``subset`` / ``paramset``
        combination.

        Parameters
        ----------
        subsets : dict, optional
            Submodel data subsets to include. If None, include all
            submodel data subsets. Default is None.
        paramsets : dict, optional
            Submodel parameter sets to include. If None, include all
            submodel parameter sets. Default is None.

        Returns
        -------
        list of tuple
            Stage submodels.

        """
        # TODO: Add examples?
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
        filtered_subsets = self.subsets
        if filtered_subsets is not None and subsets is not None:
            filtered_subsets = self.get_subset(filtered_subsets, subsets)
        filtered_paramsets = self.paramsets
        if filtered_paramsets is not None and paramsets is not None:
            filtered_paramsets = self.get_subset(filtered_paramsets, paramsets)

        # Generate all data subset/parameter set combinations
        return list(
            product(
                [None]
                if filtered_subsets is None
                else filtered_subsets.to_dict(orient="records"),
                [None]
                if filtered_paramsets is None
                else filtered_paramsets.to_dict(orient="records"),
            )  # type: ignore
        )

    @property
    def collect_after(self) -> list[str]:
        """Names of methods that collect submodel results."""
        return self._collect_after

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

    @classmethod
    def from_json(cls, config_path: Path | str, stage_name: str) -> Stage:
        """Load stage from JSON file.

        Parameters
        ----------
        config_path : Path or str
            Path to pipeline config file.
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
            del stage_config["type"]
        except KeyError:
            raise AttributeError(
                f"{pipeline_config['name']} does not contain a stage named '{stage_name}'"
            )

        stage = cls(config_path=config_path, **stage_config)
        stage.config.add_pipeline_config(pipeline_config["config"])
        return stage

    def build(
        self,
        collector: ValidationErrorCollector,
        groupby_data: Path | str | None = None,
    ) -> None:
        """Perform build-time validation and create submodels."""
        self.validate_build(collector)
        if collector.has_errors():
            return

        if not (output := self.dataif.get_path(key="output")).exists():
            output.mkdir()

        # Create data subsets
        if self.groupby is not None:
            if groupby_data is None:
                raise ValueError(
                    "groupby_data is required for groupby attribute"
                )
            self.create_subsets(groupby_data)

        # Create parameter sets
        if self.crossby is not None:
            self.create_params()

    def validate_build(
        self, collector: ValidationErrorCollector | None = None
    ) -> None:
        """Perfom build-time validation."""
        self.input.check_missing(collector=collector)

        if self.input_validation:
            for schema in self.input_validation.values():
                if isinstance(schema, Data):
                    schema.validate_metadata(kind="input", collector=collector)

        if self.output_validation:
            for schema in self.output_validation.values():
                if isinstance(schema, Data):
                    schema.validate_metadata(kind="output", collector=collector)

    def validate_run(
        self, collector: ValidationErrorCollector | None = None
    ) -> None:
        """Perfom run-time validation."""
        # TODO: add method arg
        self.input.check_exists(collector=collector)

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

    def validate_outputs(
        self, collector: ValidationErrorCollector | None = None
    ) -> None:
        """Perform post-run validation of outputs."""
        self.output.check_exists(collector=collector)

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

    def evaluate(
        self,
        method: Literal["run", "fit", "predict", "collect"],
        subsets: dict[str, Any | list[Any]] | None,
        paramsets: dict[str, Any | list[Any]] | None,
        collect: bool | None,
        backend: Literal["local", "jobmon"],
        cluster: str | None,
        resources: Path | str | dict[str, Any] | None,
        python: Path | str | None,
        collector: ValidationErrorCollector | None = None,
        **kwargs,
    ) -> None:
        """Evaluate stage method."""
        self.validate_run(collector=collector)

        if backend == "jobmon":
            from onemod.backend.jobmon_backend import evaluate_with_jobmon

            if cluster is None:
                raise ValueError("Jobmon backend requires cluster name")
            if resources is None:
                raise ValueError("Jobmon backend requires compute resources")

            evaluate_with_jobmon(
                model=self,
                method=method,
                subsets=subsets,
                paramsets=paramsets,
                collect=collect,
                cluster=cluster,
                resources=resources,
                python=python,
                **kwargs,
            )
        else:
            from onemod.backend.local_backend import evaluate_local

            evaluate_local(
                model=self,
                method=method,
                subsets=subsets,
                paramsets=paramsets,
                collect=collect,
                **kwargs,
            )

        # FIXME: only validate outputs created by method
        # self.validate_outputs(method, collector)

    def run(
        self,
        subsets: dict[str, Any | list[Any]] | None = None,
        paramsets: dict[str, Any | list[Any]] | None = None,
        collect: bool | None = None,
        backend: Literal["local", "jobmon"] = "local",
        cluster: str | None = None,
        resources: Path | str | dict[str, Any] | None = None,
        python: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Run stage.

        Parameters
        ----------
        subsets : dict, optional
            Submodel data subsets to run. If None, run all data subsets.
            Default is None.
        paramsets : dict, optional
            Submodel parameter sets to run. If None, run all parameter
            sets. Default is None.
        collect : bool, optional
            Whether to collect submodel results. If ``subsets`` and
            ``paramsets`` are both None, default is True, otherwise
            default is False.
        backend : {'local', 'jobmon'}, optional
            How to evaluate the method. Default is 'local'.
        **kwargs
            Additional keyword arguments passed to stage method.

        Jobmon Parameters
        -----------------
        cluster : str, optional
            Cluster name. Required if ``backend`` is 'jobmon'.
        resources : Path, str, or dict, optional
            Path to resources file or dictionary of compute resources.
            Required if ``backend`` is 'jobmon'.
        Python : Path or str, optional
            Path to Python environment if ``backend`` is 'jobmon'. If
            None, use sys.executable. Default is None.

        """
        self.evaluate(
            "run",
            subsets,
            paramsets,
            collect,
            backend,
            cluster,
            resources,
            python,
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

        """
        raise NotImplementedError("Subclasses must implement this method")

    def fit(
        self,
        subsets: dict[str, Any | list[Any]] | None = None,
        paramsets: dict[str, Any | list[Any]] | None = None,
        collect: bool | None = None,
        backend: Literal["local", "jobmon"] = "local",
        cluster: str | None = None,
        resources: Path | str | dict[str, Any] | None = None,
        python: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Fit stage.

        Parameters
        ----------
        subsets : dict, optional
            Submodel data subsets to fit. If None, fit all data subsets.
            Default is None.
        paramsets : dict, optional
            Submodel parameter sets to fit. If None, fit all parameter
            sets. Default is None.
        collect : bool, optional
            Whether to collect submodel results. If ``subsets`` and
            ``paramsets`` are both None, default is True, otherwise
            default is False.
        backend : {'local', 'jobmon'}, optional
            How to evaluate the method. Default is 'local'.
        **kwargs
            Additional keyword arguments passed to stage method.

        Jobmon Parameters
        -----------------
        cluster : str, optional
            Cluster name. Required if ``backend`` is 'jobmon'.
        resources : Path, str, or dict, optional
            Path to resources file or dictionary of compute resources.
            Required if ``backend`` is 'jobmon'.
        python : Path or str, optional
            Path to Python environment if ``backend`` is 'jobmon'. If
            None, use sys.executable. Default is None.

        """
        self.evaluate(
            "fit",
            subsets,
            paramsets,
            collect,
            backend,
            cluster,
            resources,
            python,
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

        """
        raise NotImplementedError(
            "Subclasses must implement this method if not skipped"
        )

    def predict(
        self,
        subsets: dict[str, Any | list[Any]] | None = None,
        paramsets: dict[str, Any | list[Any]] | None = None,
        collect: bool | None = None,
        backend: Literal["local", "jobmon"] = "local",
        cluster: str | None = None,
        resources: Path | str | dict[str, Any] | None = None,
        python: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Create stage predictions.

        Parameters
        ----------
        subsets : dict, optional
            Submodel data subsets to create predictions for. If None,
            create predictions for all data subsets. Default is None.
        paramsets : dict, optional
            Submodel parameter sets to create predictions for. If None,
            create predictions for all parameter sets. Default is None.
        collect : bool, optional
            Whether to collect submodel results. If ``subsets`` and
            ``paramsets`` are both None, default is True, otherwise
            default is False.
        backend : {'local', 'jobmon'}, optional
            How to evaluate the method. Default is 'local'.
        **kwargs
            Additional keyword arguments passed to stage method.

        Jobmon Parameters
        -----------------
        cluster : str, optional
            Cluster name. Required if ``backend`` is 'jobmon'.
        resources : Path, str, or dict, optional
            Path to resources file or dictionary of compute resources.
            Required if ``backend`` is 'jobmon'.
        python : Path or str, optional
            Path to Python environment if ``backend`` is 'jobmon'. If
            None, use sys.executable. Default is None.

        """
        self.evaluate(
            "fit",
            subsets,
            paramsets,
            collect,
            backend,
            cluster,
            resources,
            python,
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

        """
        raise NotImplementedError(
            "Subclasses must implement this method if not skipped"
        )

    def collect(self) -> None:
        """Collect stage submodel results."""
        raise NotImplementedError(
            "Subclasses must implement this method if using submodels"
            " and collect_after not empty"
        )

    def __call__(self, **input: Data | Path | str) -> Output:
        """Define stage dependencies.

        Parameters
        ----------
        **input: Data, Path, or str
            Stage input items.

        Returns
        -------
        Output
            Stage output metadata.

        """
        # TODO: Could add more description, like dataif, input keys
        self.input.update(input)

        for item_name, item_value in self.input.items.items():
            self.dataif.add_path(item_name, item_value.path, exist_ok=True)

        return self.output

    def __repr__(self) -> str:
        stage_str = f"{self.type}(name={self.name}"
        if self.groupby is not None:
            stage_str += f", groupby={self.groupby}"
        if self.crossby is not None:
            stage_str += f", crossby={self.crossby}"
        return stage_str + ")"
