"""Pipeline class."""

from __future__ import annotations

import json
import logging
from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmodulename
from pathlib import Path

from pydantic import BaseModel, computed_field

import onemod.stage as onemod_stages
from onemod.config import PipelineConfig
from onemod.serializers import deserialize, serialize
from onemod.stage import CrossedStage, GroupedStage, Stage
from onemod.validation import validation_context, ValidationErrorCollector

logger = logging.getLogger(__name__)


class Pipeline(BaseModel):
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
    groupby: set[str] | None = None
    _stages: dict[str, Stage] = {}  # set by Pipeline.add_stage

    @computed_field
    @property
    def stages(self) -> dict[str, Stage]:
        return self._stages

    @computed_field
    @property
    def dependencies(self) -> dict[str, set[str]]:
        return {stage.name: stage.dependencies for stage in self._stages}

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

    def build_dag(self) -> dict[str, list[str]]:
        """Build directed acyclic graph (DAG) from the stages and their dependencies."""
        # TODO: Placeholder until DAG class is implemented, assuming we need one
        return self.dependencies

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
        for stage, dependencies in self.dependencies.items():
            # Check for undefined dependencies
            for dep in dependencies:
                if dep not in self._stages:
                    collector.add_error(stage, "DAG validation", f"Upstream dependency '{dep}' is not defined.")

            # Check for self-dependencies
            if stage in dependencies:
                collector.add_error(stage, "DAG validation", f"Stage cannot depend on itself.")

            # Check for duplicate dependencies
            if len(dependencies) != len(set(dependencies)):
                collector.add_error(stage, "DAG validation", f"Duplicate dependencies found.")
                
    def validate_stages(self, collector: ValidationErrorCollector) -> None:
        """Validate that the specified Input, Output types between dependent stages are compatible."""
        for stage, dependencies in self.dependencies.items():
            # Dependency validation
            for dep in dependencies:
                # Check if output of upstream dependency is compatible with input of current stage
                if not self.stages[dep].output.is_compatible(self.stages[stage].input): # TODO: implement is_compatible or similar
                    collector.add_error(
                        stage,
                        "Stage validation",
                        f"Output of upstream stage '{dep}' is not compatible with input of '{stage}'."
                    )
    
    def validate(self) -> None:
        """Validate pipeline."""
        with validation_context() as collector:
            # Validate DAG structure
            self.validate_dag(collector)
            # Validate specified Input, Output types between dependent stages
            self.validate_stages(collector)
            
            if collector.has_errors():
                self.save_validation_report(collector)
                raise ValueError(f"Pipeline validation failed. See {self.directory / 'validation' / 'validation_report.json'} for details.")
        
    def save_validation_report(self, collector: ValidationErrorCollector):
        validation_dir = self.directory / 'validation'
        validation_dir.mkdir(exist_ok=True)
        report_path = validation_dir / 'validation_report.json'
        serialize(collector, report_path)
    
    def build(self) -> dict:
        """
        Assembles the Pipeline into a serializable structure.
        
        Notes
        -----
        * TODO: Include (?) OneMod, project version info
        * TODO: execution/orchestration-related metadata (sold separately?)
        * TODO: Nest the pipeline configuration within a 'config' key?
        """
        pipeline_dict = {
            "name": self.name,
            "directory": str(self.directory),
            # "config": self.config.to_dict(), # for instance
            "ids": self.config.ids,
            "obs": self.config.obs,
            "pred": self.config.pred,
            "weights": self.config.weights,
            "test": self.config.test,
            "holdouts": self.config.holdouts,
            "mtype": self.config.mtype,
            "groupby": self.groupby or [],
            "stages": {},
            "dependencies": {},
            # "execution": {}  # TODO: execution-related metadata
        }

        for stage_name, stage in self._stages.items():
            pipeline_dict["stages"][stage_name] = stage.to_dict()

        pipeline_dict["dependencies"] = {
            stage_name: dependencies
            for stage_name, dependencies in self._dependencies.items()
        }

        # Placeholder for execution-related metadata (orchestration tool, cluster settings)
        # pipeline_dict["execution"] = {
        #     "tool": "sequential",  # Placeholder for orchestration tool
        #     "cluster_name": None,  # Placeholder for cluster name
        #     "config": {}  # Placeholder for orchestration configurations
        # }

        return pipeline_dict
    
    def save(self, filepath: Path | str) -> None:
        """Save pipeline to file."""
        serialize(self.build(), filepath)

    def run(
        self,
        tool: str = "jobmon",
        config: dict | None = None,
        config_file: Path | None = None,
    ) -> None:
        """Run pipeline.

        Notes
        -----
        * These functions will handle workflow creation
        * Maybe allow subsets of the DAG to be run (e.g., predict for a
          single location)
        * TODO: consume orchestration tool config, set and validate execution options
        * TODO: run for selected stages
        * TODO: run for selected subsets or params

        """
        self.validate()
        self.build()
        self.save()
        
        # run
        match tool:
            case 'jobmon':
                # TODO: set execution-related metadata here and/or call jobmon module?
                raise NotImplementedError()
            case 'sequential':
                # PoC: simply run stages in sequence (no concurrency)
                for stage in self.get_execution_order():
                    self.stages[stage].validate_inputs()
                    self.stages[stage].run()
                    self.stages[stage].validate_outputs()
            case _:
                raise ValueError(f"Unsupported execution tool: {tool}")

    def resume(self) -> None:
        """Resume pipeline."""
        raise NotImplementedError()

    def fit(self) -> None:
        """Fit pipeline model.

        Notes
        -----
        * Skip any stage that has 'fit' in `skip_if` attribute

        """
        raise NotImplementedError()

    def predict(self) -> None:
        """Predict pipeline model.

        Notes
        -----
        * Skip any stage that has 'predict' in `skip_if` attribute

        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
