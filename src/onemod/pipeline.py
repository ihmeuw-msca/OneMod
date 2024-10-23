"""Pipeline class."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Literal

from pplkit.data.interface import DataInterface
from pydantic import BaseModel, computed_field, validate_call

from onemod.config import PipelineConfig
from onemod.serializers import serialize
from onemod.stage import ModelStage, Stage
from onemod.validation import ValidationErrorCollector, handle_error

logger = logging.getLogger(__name__)


class Pipeline(BaseModel):
    """Pipeline class.

    Parameters
    ----------
    name : str
        Pipeline name.
    config : PipelineConfig
        Pipeline configuration.
    directory : Path
        Experiment directory.
    data : Path or None, optional
        Input data used to create data subsets. Required for pipeline or
        stage `groupby` attribute. Default is None.
    groupby : set of str or None, optional
        Column names used to create data subsets. Default is None.

    """

    name: str
    config: PipelineConfig
    directory: Path
    data: Path | None = None
    groupby: set[str] | None = None
    _dataif: DataInterface  # set by model_post_init, build
    _stages: dict[str, Stage] = {}  # set by add_stage

    @property
    def dataif(self) -> DataInterface:
        return self._dataif

    @computed_field
    @property
    def dependencies(self) -> dict[str, set[str]]:
        return {
            stage.name: stage.dependencies for stage in self.stages.values()
        }

    @computed_field
    @property
    def stages(self) -> dict[str, Stage]:
        return self._stages

    def model_post_init(self, *args, **kwargs) -> None:
        if not self.directory.exists():
            self.directory.mkdir(parents=True)
        self._dataif = DataInterface(directory=self.directory)

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
            raise ValueError(f"stage '{stage.name}' already exists")

        stage.config.inherit(self.config)
        self._stages[stage.name] = stage

    def get_execution_order(self) -> list[str]:
        """
        Return topologically sorted order of stages, ensuring no cycles.
        Uses Kahn's algorithm to find the topological order of the stages.
        """
        reverse_graph = {stage: [] for stage in self.dependencies}
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

    def build(self) -> None:
        """Assemble pipeline, perform build-time validation, and save to JSON."""
        collector = ValidationErrorCollector()

        for stage in self.stages.values():
            stage.validate_build(collector)

        self.validate_dag(collector)

        if collector.has_errors():
            self.save_validation_report(collector)
            collector.raise_errors()

        config_path = self.directory / (self.name + ".json")
        self.dataif.add_dir("config", config_path)

        for stage in self.stages.values():
            self.dataif.add_dir(stage.name, self.directory / stage.name)
            stage.set_dataif(config_path)

            # Create data subsets
            if isinstance(stage, ModelStage):
                if self.groupby is not None:
                    if stage.groupby is None:
                        stage.groupby = self.groupby
                    else:
                        stage.groupby.update(self.groupby)
                if stage.groupby:
                    if self.data is None:
                        raise AttributeError("data is required for groupby")
                    stage.create_stage_subsets(self.data)

            # Create parameter sets
            if isinstance(stage, ModelStage):
                if stage.config.crossable_params:
                    stage.create_stage_params()

        self.to_json(config_path)

    def save_validation_report(
        self, collector: ValidationErrorCollector
    ) -> None:
        validation_dir = self.directory / "validation"
        validation_dir.mkdir(exist_ok=True)
        report_path = validation_dir / "validation_report.json"
        serialize(collector.errors, report_path)

    @validate_call
    def evaluate(
        self,
        method: Literal["run", "fit", "predict"] = "run",
        backend: Literal["local", "jobmon"] = "local",
        build: bool = True,
        **kwargs,
    ) -> None:
        """Evaluate pipeline method.

        Parameters
        ----------
        method : str, optional
            Name of method to evaluate. Default is 'run'.
        backend : str, optional
            How to evaluate the method. Default is 'local'.

        Other Parameters
        ----------------
        cluster : str, optional
            Cluster name. Required if `backend` is 'jobmon'.
        resources : Path or str, optional
            Path to resources yaml file. Required if `backend` is
            'jobmon'.

        TODO: Add options to run subset of stages
        TODO: Add options to run subset of IDs

        """
        if build:
            self.build()
        if backend == "jobmon":
            from onemod.backend import evaluate_with_jobmon

            evaluate_with_jobmon(model=self, method=method, **kwargs)
        else:
            from onemod.backend import evaluate_local

            evaluate_local(model=self, method=method)

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
            f"{type(self).__name__}({self.name},"
            f" stages={list(self.stages.values())})"
        )
