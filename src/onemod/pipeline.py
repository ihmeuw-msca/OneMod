"""Pipeline class."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from onemod.config import Config
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
    directory : Path or str
        Path to pipeline directory.
    config : Config or dict, optional
        Pipeline configuration.
    groupby_data : Path or str, optional
        Path to data file used to create stage data subsets. Must
        contain all columns included in stage ``groupby`` attributes.

    """

    name: str
    directory: Path
    config: Config = Config()
    groupby_data: Path | None = None
    _stages: dict[str, Stage] = {}

    @computed_property
    def stages(self) -> dict[str, Stage]:
        """Pipeline stages."""
        return self._stages

    @computed_property
    def dependencies(self) -> dict[str, list[str]]:
        """Stage dependencies."""
        return {
            stage.name: stage.dependencies for stage in self.stages.values()
        }

    @classmethod
    def from_json(cls, config_path: Path | str) -> Pipeline:
        """Load pipeline from JSON file.

        Parameters
        ----------
        config_path : Path or str
            Path to pipeline config file.

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
        config_path : Path or str, optional
            Where to save config file. If None, file is saved at
            pipeline.directory / (pipeline.name + ".json"). Default is
            None.

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
        stage.set_dataif(self.directory / (self.name + ".json"))
        stage.set_output()

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

        """
        # TODO: What if stages have a gap? For example, pipeline has
        # Rover -> SPxMod -> KReg, but `stages` only includes Rover and
        # Kreg. KReg will be run on outdated SPxMod results (if they
        # exist).
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

    def build(self) -> None:
        """Assemble pipeline, perform build-time validation, and save to JSON."""
        collector = ValidationErrorCollector()

        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        for stage in self.stages.values():
            stage.build(collector, self.groupby_data)

        self.validate_dag(collector)

        if collector.has_errors():
            self.save_validation_report(collector)
            collector.raise_errors()

        self.to_json(self.directory / (self.name + ".json"))

    def save_validation_report(
        self, collector: ValidationErrorCollector
    ) -> None:
        validation_dir = self.directory / "validation"
        validation_dir.mkdir(exist_ok=True)
        report_path = validation_dir / "validation_report.json"
        serialize(collector.errors, report_path)  # type: ignore[arg-type]

    def evaluate(
        self,
        method: Literal["run", "fit", "predict", "collect"],
        stages: list[str] | None,
        backend: Literal["local", "jobmon"],
        cluster: str | None,
        resources: Path | str | dict[str, Any] | None,
        python: Path | str | None,
        **kwargs,
    ) -> None:
        """Evaluate pipeline method."""
        if backend == "jobmon":
            from onemod.backend.jobmon_backend import evaluate_with_jobmon

            if cluster is None:
                raise ValueError("Jobmon backend requires cluster name")
            if resources is None:
                raise ValueError("Jobmon backend requires compute resources")

            evaluate_with_jobmon(
                model=self,
                method=method,
                stages=stages,
                cluster=cluster,
                resources=resources,
                python=python,
                **kwargs,
            )
        else:
            from onemod.backend.local_backend import evaluate_local

            evaluate_local(model=self, method=method, stages=stages, **kwargs)

    def run(
        self,
        stages: list[str] | None = None,
        backend: Literal["local", "jobmon"] = "local",
        cluster: str | None = None,
        resources: Path | str | dict[str, Any] | None = None,
        python: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Run pipeline stages.

        All stage submodels are run and submodel results are collected.

        Parameters
        ----------
        stages : list of str, optional
            Names of stages to run. If None, run all pipeline stages.
            Default is None.
        backend : {'local', 'jobmon'}, optional
            How to evaluate the method. Default is 'local'.
        **kwargs
            Additional keyword arguments passed to stage methods. Use
            format ``stage={arg_name: arg_value}``.

        Jobmon Parameters
        ----------------
        cluster : str, optional
            Cluster name. Required if ``backend`` is 'jobmon'.
        resources : Path, str, or dict, optional
            Path to resources file or dictionary of compute resources.
            Required if ``backend`` is 'jobmon'.
        python : Path, or str, optional
            Path to Python environment if ``backend`` is 'jobmon'. If
            None, use sys.executable. Default is None.

        """
        self.evaluate(
            "run", stages, backend, cluster, resources, python, **kwargs
        )

    def fit(
        self,
        stages: list[str] | None = None,
        backend: Literal["local", "jobmon"] = "local",
        cluster: str | None = None,
        resources: Path | str | dict[str, Any] | None = None,
        python: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Fit pipeline stages.

        All stage submodels are fit and submodel results are collected.

        Parameters
        ----------
        stages : list of str, optional
            Names of stages to fit. If None, fit entire pipeline.
            Default is None.
        backend : {'local', 'jobmon'}, optional
            How to evaluate the method. Default is 'local'.
        **kwargs
            Additional keyword arguments passed to stage methods. Use
            format ``stage={arg_name: arg_value}``.

        Jobmon Parameters
        -----------------
        cluster : str, optional
            Cluster name. Required if ``backend`` is 'jobmon'.
        resources : Path, str, or dict, optional
            Path to resources file or dictionary of compute resources.
            Required if ``backend`` is 'jobmon'.
        python : Path, or str, optional
            Path to Python environment if ``backend`` is 'jobmon'. If
            None, use sys.executable. Default is None.

        """
        self.evaluate(
            "fit", stages, backend, cluster, resources, python, **kwargs
        )

    def predict(
        self,
        stages: list[str] | None = None,
        backend: Literal["local", "jobmon"] = "local",
        cluster: str | None = None,
        resources: Path | str | dict[str, Any] | None = None,
        python: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Create predictions for pipeline stages.

        Predictions are made for all stage submodels and submodel
        results are collected.

        Parameters
        ----------
        stages : list of str, optional
            Names of stages to create predictions for. If None, create
            predictions for the entire pipeline. Default is None.
        backend : {'local', 'jobmon'}, optional
            How to evaluate the method. Default is 'local'.
        **kwargs
            Additional keyword arguments passed to stage methods. Use
            format ``stage={arg_name: arg_value}``.

        Jobmon Parameters
        -----------------
        cluster : str, optional
            Cluster name. Required if ``backend`` is 'jobmon'.
        resources : Path, str, or dict, optional
            Path to resources file or dictionary of compute resources.
            Required if ``backend`` is 'jobmon'.
        python : Path, or str, optional
            Path to Python environment if ``backend`` is 'jobmon'. If
            None, use sys.executable. Default is None.

        """
        self.evaluate(
            "predict", stages, backend, cluster, resources, python, **kwargs
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name},"
            f" stages={list(self.stages.values())})"
        )
