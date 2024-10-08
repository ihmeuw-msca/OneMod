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
from onemod.stage import CrossedStage, GroupedStage, Stage

logger = logging.getLogger(__name__)


class Pipeline(BaseModel):
    """
    Notes
    -----
    * `data` is raw input data used to figure out grouped stage subsets
       when initializing the pipeline; stages will get their data from
       other stages (with the exception of any preprocessing stages)
    * `data` is only required if using grouped stages
    * TODO: Define dependencies between stages
    * TODO: DAG creation, validation
    * TODO: Run locally or on Jobmon

    """

    name: str
    config: PipelineConfig
    directory: Path
    data: Path | None = None
    groupby: set[str] = set()
    _stages: dict[str, Stage] = {}  # set by Pipeline.add_stage
    _dependencies: dict[str, list[str]] = {}

    @computed_field
    @property
    def stages(self) -> dict[str, Stage]:
        return self._stages

    @computed_field
    @property
    def dependencies(self) -> dict[str, list[str]]:
        return self._dependencies

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
        with open(filepath, "r") as f:
            config = json.load(f)

        stages = config.pop("stages", {})
        dependencies = config.pop("dependencies", {})

        pipeline = cls(**config)

        if stages:
            pipeline.add_stages(
                [
                    pipeline.stage_from_json(
                        filepath, stage, from_pipeline=True
                    )
                    for stage in stages
                ],
                dependencies,
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
        with open(filepath, "w") as f:
            f.write(
                self.model_dump_json(
                    indent=4, exclude_none=True, serialize_as_any=True
                )
            )

    def add_stages(
        self, stages: list[Stage | str], dependencies: dict[str, list[str]] = {}
    ) -> None:
        """Add stages and dependencies to pipeline."""
        for stage in stages:
            self.add_stage(stage, dependencies.get(stage.name, []))

    def add_stage(self, stage: Stage, dependencies: list[str] = []) -> None:
        """Add stage and dependencies to pipeline."""
        self._add_stage(stage)
        self._dependencies[stage.name] = []
        if dependencies:
            self._add_dependencies(stage, dependencies)

    def _add_stage(self, stage: Stage) -> None:
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
            stage.groupby.update(self.groupby)
            stage.create_stage_subsets(self.data)

        # Create parameter sets
        if isinstance(stage, CrossedStage):
            if stage.config.crossable_params:
                stage.create_stage_params()

        self._stages[stage.name] = stage

    def _add_dependencies(self, stage: Stage, dependencies: list[str]) -> None:
        """Add stage dependencies to pipeline."""
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(
                f"Duplicate dependencies found for stage '{stage.name}'"
            )
        for dep in dependencies:
            if dep not in self.stages:
                raise ValueError(f"Dependency '{dep}' not found in pipeline.")
            self._dependencies[stage.name].append(dep)

    @classmethod
    def stage_from_json(
        cls,
        filepath: Path | str,
        name: str | None = None,
        from_pipeline: bool = False,
    ) -> Stage:
        """Load stage from JSON file.

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
        return self._dependencies

    def validate_dag(self):
        """Validate that the DAG structure is correct."""
        for stage, dependencies in self._dependencies.items():
            # Check for undefined dependencies
            for dep in dependencies:
                if dep not in self._stages:
                    raise ValueError(
                        f"Stage '{dep}' is not defined, but '{stage}' depends on it."
                    )

            # Check for self-dependencies
            if stage in dependencies:
                raise ValueError(f"Stage '{stage}' cannot depend on itself.")

            # Check for duplicate dependencies
            if len(dependencies) != len(set(dependencies)):
                raise ValueError(
                    f"Duplicate dependencies found for stage '{stage}'."
                )

    def get_execution_order(self) -> list[str]:
        """
        Return topologically sorted order of stages, ensuring no cycles.
        Uses Kahn's algorithm to find the topological order of the stages.
        """
        reverse_graph = {stage: [] for stage in self._dependencies}
        in_degree = {stage: 0 for stage in self._dependencies}
        for stage, deps in self._dependencies.items():
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
        if len(topological_order) != len(self._dependencies):
            unvisited = set(self._dependencies) - visited
            raise ValueError(
                f"Cycle detected! Unable to process the following stages: {unvisited}"
            )

        return topological_order

    def run(self) -> None:
        """Run pipeline.

        Notes
        -----
        * These functions will handle workflow creation
        * Maybe allow subsets of the DAG to be run (e.g., predict for a
          single location)
        * TODO: run for selected stages
        # TODO: run for selected subsets or params

        """
        raise NotImplementedError()

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
