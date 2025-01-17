"""Pipeline class."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, validate_call

from onemod.config import Config
from onemod.dtypes.unique_sequence import UniqueList, update_unique_list
from onemod.serialization import serialize
from onemod.stage import Stage
from onemod.utils.decorators import computed_property
from onemod.validation import ValidationErrorCollector, handle_error

logger = logging.getLogger(__name__)


class Pipeline(BaseModel):
    """Pipeline class.

    Parameters
    ----------
    name : str
        Pipeline name.
    config : Config, optional
        Pipeline configuration.
    directory : Path
        Experiment directory.
    groupby : list of str, optional
        Column names used to create data subsets. Default is an empty
        list.
    groupby_data : Path or None, optional
        Path to the data file used for creating data subsets. Default is
        None. Required when specifying pipeline or stage `groupby`
        attribute. All columns specified in pipeline or stage `groupby`
        must be present in `groupby_data`.

    """

    name: str
    config: Config = Config()
    directory: Path
    groupby: UniqueList[str] = []
    groupby_data: Path | None = None
    id_subsets: dict[str, list[Any]] | None = None
    _stages: dict[str, Stage] = {}  # set by add_stage

    @computed_property
    def dependencies(self) -> dict[str, list[str]]:
        return {
            stage.name: stage.dependencies for stage in self.stages.values()
        }

    @computed_property
    def stages(self) -> dict[str, Stage]:
        return self._stages

    def model_post_init(self, *args, **kwargs) -> None:
        if len(self.groupby) > 0 and self.groupby_data is None:
            raise AttributeError(
                "groupby_data is required for groupby attribute"
            )
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

    @classmethod
    def from_json(cls, config_path: Path | str) -> Pipeline:
        """Load pipeline from JSON file.

        Parameters
        ----------
        config_path : Path or str
            Path to config file.

        Returns
        -------
        Pipeline
            Pipeline instance.

        """
        with open(config_path, "r") as file:
            config = json.load(file)

        stages = config.pop("stages", {})

        pipeline = cls(**config)

        if stages:
            from onemod.main import load_stage

            pipeline.add_stages(
                [load_stage(config_path, stage) for stage in stages]
            )

        return pipeline

    def to_json(self, config_path: Path | str | None = None) -> None:
        """Save pipeline as JSON file.

        Parameters
        ----------
        config_path : Path, str, or None, optional
            Where to save config file. If None, file is saved at
            pipeline.directory / (pipeline.name + ".json").
            Default is None.

        """
        config_path = config_path or self.directory / (self.name + ".json")
        with open(config_path, "w") as file:
            file.write(
                self.model_dump_json(
                    indent=2, exclude_none=True, serialize_as_any=True
                )
            )

    def add_stages(self, stages: list[Stage]) -> None:
        """Add stages to pipeline.

        Parameters
        ----------
        stages : list of Stage
            Stages to add to the pipeline.

        """
        for stage in stages:
            self.add_stage(stage)

    def add_stage(self, stage: Stage) -> None:
        """Add stage to pipeline.

        Parameters
        ----------
        stage : Stage
            Stage to add to the pipeline.

        """
        if stage.name in self.stages:
            raise ValueError(f"Stage '{stage.name}' already exists")

        stage.config.add_pipeline_config(self.config)
        self._stages[stage.name] = stage

    def get_execution_order(self, stages: list[str] | None = None) -> list[str]:
        """Get stages sorted in execution order.

        Use Kahn's algorithm to find the topoligical order of the
        stages, ensuring no cycles.

        Parameters
        ----------
        stages: list of str, optional
            Name of stages to sort. If None, sort all pipeline stages.
            Default is None.

        Returns
        -------
        list of str
            Stages sorted in execution order.

        Raises
        ------
        ValueError
            If cycle detected in DAG.

        TODO: What if stages have a gap? For example, pipeline has
        Rover -> SPxMod -> KReg, but `stages` only includes Rover and
        Kreg. KReg will be run on outdated SPxMod results (if they
        exist).

        """
        reverse_graph: dict[str, list[str]] = {
            stage: [] for stage in self.dependencies
        }
        in_degree = {stage: 0 for stage in self.dependencies}
        for stage, deps in self.dependencies.items():
            for dep in deps:
                reverse_graph[dep].append(stage)
                in_degree[stage] += 1

        queue = deque([stage for stage, deg in in_degree.items() if deg == 0])
        topological_order = []
        visited = set()

        while queue:
            stage = queue.popleft()
            topological_order.append(stage)
            visited.add(stage)

            # Reduce the in-degree of downstream stages
            for downstream_dep in reverse_graph[stage]:
                in_degree[downstream_dep] -= 1
                if in_degree[downstream_dep] == 0:
                    queue.append(downstream_dep)

        # If there is a cycle, the topological order will not include all stages
        if len(topological_order) != len(self.dependencies):
            unvisited = set(self.dependencies) - visited
            raise ValueError(
                f"Cycle detected! Unable to process the following stages: {unvisited}"
            )

        if stages:
            return [stage for stage in topological_order if stage in stages]
        return topological_order

    def validate_dag(self, collector: ValidationErrorCollector) -> None:
        """Validate that the DAG structure is correct."""
        for stage_name, dependencies in self.dependencies.items():
            for dep in dependencies:
                if dep not in self._stages:
                    handle_error(
                        stage_name,
                        "DAG validation",
                        ValueError,
                        f"Upstream dependency '{dep}' is not defined.",
                        collector,
                    )

                if dep == stage_name:
                    handle_error(
                        stage_name,
                        "DAG validation",
                        ValueError,
                        "Stage cannot depend on itself.",
                        collector,
                    )

            if len(dependencies) != len(set(dependencies)):
                handle_error(
                    stage_name,
                    "DAG validation",
                    ValueError,
                    "Duplicate dependencies found.",
                    collector,
                )

        try:
            self.get_execution_order()
        except ValueError as e:
            handle_error(
                "Pipeline", "DAG validation", ValueError, str(e), collector
            )

    def build(self, id_subsets: dict[str, list[Any]] | None = None) -> None:
        """Assemble pipeline, perform build-time validation, and save to JSON."""
        self.id_subsets = id_subsets
        collector = ValidationErrorCollector()

        if self.id_subsets is not None:
            invalid_keys = set(self.id_subsets) - set(self.groupby or [])
            if invalid_keys:
                raise ValueError(
                    f"id_subsets keys {invalid_keys} do not match groupby columns {self.groupby}"
                )

        for stage in self.stages.values():
            stage.validate_build(collector)

        self.validate_dag(collector)

        if collector.has_errors():
            self.save_validation_report(collector)
            collector.raise_errors()

        config_path = self.directory / (self.name + ".json")
        for stage in self.stages.values():
            stage.set_dataif(config_path)

            # Create data subsets
            if len(self.groupby) > 0:
                if len(stage.groupby) == 0:
                    stage.groupby = self.groupby
                else:
                    stage.groupby = update_unique_list(
                        self.groupby, stage.groupby
                    )
            if stage.groupby:
                if self.groupby_data is None:
                    raise AttributeError(
                        "groupby_data is required for groupby attribute"
                    )
                stage.dataif.add_path("groupby_data", self.groupby_data)
                stage.create_stage_subsets(
                    data_key="groupby_data", id_subsets=self.id_subsets
                )

            # Create parameter sets
            if len(stage.crossby) > 0:
                stage.create_stage_params()

        self.to_json(config_path)

    def save_validation_report(
        self, collector: ValidationErrorCollector
    ) -> None:
        validation_dir = self.directory / "validation"
        validation_dir.mkdir(exist_ok=True)
        report_path = validation_dir / "validation_report.json"
        serialize(collector.errors, report_path)  # type: ignore[arg-type]

    @validate_call
    def evaluate(
        self,
        method: Literal["run", "fit", "predict", "collect"] = "run",
        stages: UniqueList[str] | None = None,
        backend: Literal["local", "jobmon"] = "local",
        build: bool = True,
        id_subsets: dict[str, list[Any]] | None = None,
        **kwargs,
    ) -> None:
        """Evaluate pipeline method.

        Parameters
        ----------
        method : str, optional
            Name of method to evaluate. Default is 'run'.
        stages : list of str, optional
            Names of stages to evaluate. Default is None.
            If None, evaluate entire pipeline.
        backend : str, optional
            How to evaluate the method. Default is 'local'.
        build : bool, optional
            Whether to build the pipeline before evaluation.
            Default is True.
        id_subsets : dict of str: list of Any, optional

        Other Parameters
        ----------------
        cluster : str, optional
            Cluster name. Required if `backend` is 'jobmon'.
        resources : Path, str, or dict, optional
            Dictionary of compute resources or path to resources file.
            Required if `backend` is 'jobmon'.

        """
        if build:
            self.build(id_subsets=id_subsets)

        if backend == "jobmon":
            from onemod.backend.jobmon_backend import evaluate
        else:
            from onemod.backend.local_backend import evaluate

        evaluate(model=self, method=method, stages=stages, **kwargs)

    def run(
        self,
        backend: Literal["local", "jobmon"] = "local",
        build: bool = True,
        **kwargs,
    ) -> None:
        """Run pipeline."""
        self.evaluate(method="run", backend=backend, build=build, **kwargs)

    def fit(
        self,
        backend: Literal["local", "jobmon"] = "local",
        build: bool = True,
        **kwargs,
    ) -> None:
        """Fit pipeline model."""
        self.evaluate(method="fit", backend=backend, build=build, **kwargs)

    def predict(
        self,
        backend: Literal["local", "jobmon"] = "local",
        build: bool = True,
        **kwargs,
    ) -> None:
        """Predict pipeline model."""
        self.evaluate(method="predict", backend=backend, build=build, **kwargs)

    def resume(self) -> None:
        """Resume pipeline."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name},"
            f" stages={list(self.stages.values())})"
        )
