"""Pipeline class."""

from __future__ import annotations

import json
import logging
from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pydantic import BaseModel, computed_field

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
    def stages(self) -> set[str]:
        return self._stages

    def model_post_init(self, *args, **kwargs) -> None:
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

    @classmethod
    def from_json(cls, filepath: Path | str) -> Pipeline:
        """Load pipeline object from JSON file."""
        with open(filepath, "r") as f:
            pipeline_json = json.load(f)

        stages = pipeline_json.pop("stages", None)
        dependencies = pipeline_json.pop("dependencies", {})

        pipeline = cls(**pipeline_json)

        if stages is not None:
            pipeline.add_stages(stages)
        if dependencies:
            pipeline.add_stages_with_dependencies(dependencies)

        return pipeline

    def to_json(self, filepath: Path | str | None = None) -> None:
        """Save pipeline object as JSON file."""
        filepath = filepath or self.directory / (self.name + ".json")
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))

    def add_stages(
        self, stages: list[Stage | str], filepath: Path | str | None = None
    ) -> None:
        """Add stages to pipeline."""
        for stage in stages:
            self.add_stage(stage, filepath)

    def add_stages_with_dependencies(
        self, stages_with_deps: dict[str, list[str]]
    ) -> None:
        """Add multiple stages with their corresponding dependencies."""
        for stage_name, dependencies in stages_with_deps.items():
            self.add_stage(stage_name, dependencies=dependencies)

    def add_stage(
        self,
        stage: Stage | str,
        filepath: Path | str | None = None,
        dependencies: list[str] = [],
    ) -> None:
        """Add stage to pipeline."""
        if isinstance(stage, str):
            stage = self._stage_from_json(stage, filepath)

        self._add_stage(stage)

        if dependencies:
            if len(dependencies) != len(set(dependencies)):
                raise ValueError(
                    f"Duplicate dependencies found for stage '{stage.name}'"
                )
            for dep in dependencies:
                if dep not in self.stages:
                    raise ValueError(
                        f"Dependency '{dep}' not found in pipeline."
                    )
                self._dependencies[stage.name] = self._dependencies.get(
                    stage.name, []
                ) + [dep]

    def _add_stage(self, stage: Stage) -> None:
        """Add stage object to pipeline."""
        if stage.name in self.stages:
            raise ValueError(f"stage '{stage.name}' already exists")
        stage.config.update(self.config)
        stage.directory = self.directory / stage.name
        if isinstance(stage, GroupedStage):
            if self.data is None:
                raise ValueError("data is required for GroupedStage")
            stage.groupby.update(self.groupby)
            stage.create_stage_subsets(self.data)
        if isinstance(stage, CrossedStage):
            if stage.config.crossable_params:
                stage.create_stage_params()
        self._stages[stage.name] = stage
        self._dependencies[stage.name] = []

    def _stage_from_json(self, stage: str, filepath: Path | str) -> Stage:
        """Load stage object from JSON.

        Notes
        -----
        * There may be an easier way to load built-in stages

        """
        if stage in self.stages:
            raise ValueError(f"stage '{stage}' already exists")
        with open(filepath, "r") as f:
            stage_json = json.load(f)
        spec = spec_from_file_location(stage_json["module"])
        module = module_from_spec(spec)
        spec.loader.exec_module(module_from_spec(spec))
        stage_class = module.__getattribute__(stage_json["type"])
        return stage_class(**stage_json)

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

        # Check for isolated nodes
        # TODO: are there cases where isolated nodes would be valid for OneMod?
        all_stages = set(self._stages.keys())
        dependent_stages = set(
            stage for deps in self._dependencies.values() for stage in deps
        )
        isolated_stages = (
            all_stages - dependent_stages - set(self._dependencies.keys())
        )
        if isolated_stages:
            logger.warning(
                f"The following stages are isolated and not part of the DAG: {isolated_stages}"
            )

    def get_execution_order(self) -> list[str]:
        """
        Return topologically sorted order of stages, ensuring no cycles.
        Uses of Kahn's algorithm to find the topological order of the stages.
        """
        # Track in-degrees of all nodes
        in_degree = {stage: 0 for stage in self._stages}

        # Populate in-degree based on dependencies
        for stage, dependencies in self._dependencies.items():
            for dep in dependencies:
                in_degree[stage] += 1

        # Use a deque to process nodes with zero in-degree (no dependencies)
        queue = deque([stage for stage, deg in in_degree.items() if deg == 0])
        topological_order = []
        rec_stack = set()
        visited = set()

        while queue:
            stage = queue.popleft()
            topological_order.append(stage)
            visited.add(stage)

            # Reduce the in-degree of dependent stages
            for dep in self._dependencies.get(stage, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

            # Track current processing path for cycle detection
            rec_stack.add(stage)

            # Detect cycles in dependent stages
            for dep in self._dependencies.get(stage, []):
                if dep in rec_stack:
                    raise ValueError(
                        f"Cycle detected! The cycle involves: {list(rec_stack)}"
                    )

            rec_stack.remove(stage)

        # If there is a cycle, the topological order will not include all stages
        if len(topological_order) != len(self._stages):
            unvisited = set(self._stages) - visited
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
