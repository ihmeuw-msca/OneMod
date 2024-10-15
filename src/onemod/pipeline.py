"""Pipeline class."""

from __future__ import annotations

import json
import logging
from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmodulename
from itertools import product
from pathlib import Path
from typing import Literal, Set

from pydantic import computed_field, field_serializer, validate_call

import onemod.stage as onemod_stages
from onemod.base_models import SerializableModel
from onemod.config import PipelineConfig
from onemod.serializers import deserialize, serialize
from onemod.stage import CrossedStage, GroupedStage, Stage
from onemod.validation import ValidationErrorCollector, handle_error

logger = logging.getLogger(__name__)


class Pipeline(SerializableModel):
    """Pipeline class.

    Attributes
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
    groupby : set[str] or None, optional
        ID names used to create data subsets. Default is None.

    """

    name: str
    config: PipelineConfig
    directory: Path  # TODO: replace with DataInterface
    data: Path | None = None
    groupby: Set[str] | None = None
    _stages: dict[str, Stage] = {}  # set by Pipeline.add_stage

    @computed_field
    @property
    def stages(self) -> dict[str, Stage]:
        return self._stages

    @computed_field
    @property
    def dependencies(self) -> dict[str, set[str]]:
        return {stage.name: stage.dependencies for stage in self.stages.values()}
    
    @field_serializer('groupby')
    def serialize_groupby(self, value: Set[str] | None, info) -> list[str] | None:
        return list(value) if value is not None else None

    def model_post_init(self, *args, **kwargs) -> None:
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

    @classmethod
    def from_json(cls, filepath: Path | str) -> Pipeline:
        """Load pipeline from JSON file.

        Parameters
        ----------
        filepath : Path or str
            Path to config file.

        Returns
        -------
        Pipeline
            Pipeline instance.

        """
        config = deserialize(filepath)

        stages = config.pop("stages", {})

        pipeline = cls(**config)

        if stages:
            pipeline.add_stages(
                [
                    pipeline.stage_from_json(
                        filepath, stage, from_pipeline=True
                    )
                    for stage in stages
                ]
            )

        return pipeline

    def to_json(self, filepath: Path | str | None = None) -> None:
        """Save pipeline as JSON file.

        Parameters
        ----------
        filepath : Path, str, or None, optional
            Where to save config file. If None, file is saved at
            pipeline.directory / (pipeline.name + ".json").
            Default is None.

        """
        filepath = filepath or self.directory / (self.name + ".json")
        serialize(self, filepath)

    def add_stages(self, stages: list[Stage]) -> None:
        """Add stages to pipeline.

        Parameters
        ----------
        stages : list[Stage]
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

        Notes
        -----
        * Maybe move most of this into pipeline.compile?
        * Some steps may be unnecessary if stage loaded from previously
          compiled config (update config, set directory, create subsets
          and params)

        """
        if stage.name in self.stages:
            raise ValueError(f"stage '{stage.name}' already exists")
        stage.config.update(self.config)
        stage.directory = self.directory / stage.name

        # Create data subsets
        if isinstance(stage, GroupedStage):
            if self.data is None:
                raise AttributeError("data field is required for GroupedStage")
            if self.groupby is not None:
                stage.groupby.update(self.groupby)
            stage.create_stage_subsets(self.data)

        # Create parameter sets
        if isinstance(stage, CrossedStage):
            if stage.config.crossable_params:
                stage.create_stage_params()

        self._stages[stage.name] = stage

    @classmethod
    def stage_from_json(
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

        """
        with open(filepath, "r") as f:
            config = json.load(f)
        if from_pipeline:
            config = config["stages"][name]
        if hasattr(onemod_stages, stage_type := config["type"]):
            stage_class = getattr(onemod_stages, stage_type)
        else:  # custom stage
            module_path = Path(config["module"])
            spec = spec_from_file_location(
                getmodulename(module_path), module_path
            )
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            stage_class = getattr(module, stage_type)
        return stage_class.from_json(filepath, name, from_pipeline)

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
                    handle_error(stage_name, "DAG validation", ValueError,
                                 f"Upstream dependency '{dep}' is not defined.", collector)

                if dep == stage_name:
                    handle_error(stage_name, "DAG validation", ValueError,
                                "Stage cannot depend on itself.", collector)

            if len(dependencies) != len(set(dependencies)):
                handle_error(stage_name, "DAG validation", ValueError,
                                "Duplicate dependencies found.", collector)
                
        try:
            self.get_execution_order()
        except ValueError as e:
            handle_error("Pipeline", "DAG validation", ValueError, str(e), collector)
    
    def build(self) -> None:
        """Assemble the pipeline, perform build-time validation, and save it to JSON."""
        collector = ValidationErrorCollector()

        for stage in self.stages.values():
            stage.validate_build(collector)

        self.validate_dag(collector)

        if collector.has_errors():
            self.save_validation_report(collector)
            collector.raise_errors()

        pipeline_dict = self.model_dump()
        pipeline_dict["stages"] = {}
        for stage_name, stage in self.stages.items():
            pipeline_dict["stages"][stage_name] = stage.model_dump()
        pipeline_dict["dependencies"] = {
            stage_name: list(dependencies)
            for stage_name, dependencies in self.dependencies.items()
        }

        serialize(pipeline_dict, self.directory / f"{self.name}.json")
        
    def save_validation_report(self, collector: ValidationErrorCollector) -> None:
        validation_dir = self.directory / 'validation'
        validation_dir.mkdir(exist_ok=True)
        report_path = validation_dir / 'validation_report.json'
        serialize(collector.errors, report_path)

    @validate_call
    def evaluate(
        self,
        method: Literal["run", "fit", "predict"] = "run",
        backend: Literal["local", "jobmon"] = "local",
        *args,
        **kwargs,
    ) -> None:
        """Evaluate pipeline method.

        Parameters
        ----------
        method : str, optional
            Name of method to evaluate. Default is 'run'.
        backend : str, optional
            How to evaluate the method. Default is 'local'.

        TODO: Add options to run subset of stages
        TODO: Add options to run subset of IDs

        """
        if backend == "jobmon":
            raise NotImplementedError("Jobmon backend not implemented yet.")
        else:
            for stage in self.stages.values():
                if method not in stage._skip_if:
                    subset_ids = getattr(stage, "subset_ids", None)
                    param_ids = getattr(stage, "param_ids", None)
                    if subset_ids is not None or param_ids is not None:
                        for subset_id, param_id in product(
                            subset_ids or [None], param_ids or [None]
                        ):
                            stage.evaluate(
                                method=method,
                                subset_id=subset_id,
                                param_id=param_id,
                            )
                        stage.collect()
                    else:
                        stage.evaluate(method=method)

    def run(self) -> None:
        """Run pipeline."""
        self.evaluate(method="run")

    def fit(self) -> None:
        """Fit pipeline model."""
        self.evaluate(method="fit")

    def predict(self) -> None:
        """Predict pipeline model."""
        self.evaluate(method="predict")

    def resume(self) -> None:
        """Resume pipeline."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
