"""Pipeline class."""

from __future__ import annotations

import fire
import json
import logging
from collections import deque

from pathlib import Path

from pydantic import BaseModel, computed_field

import onemod
from onemod.config import PipelineConfig
from onemod.stage import CrossedStage, GroupedStage, Stage

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
        return {
            stage.name: stage.dependencies for stage in self.stages.values()
        }

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

        pipeline = cls(**config)

        if stages:
            pipeline.add_stages(
                [
                    onemod.load_stage(filepath, stage, from_pipeline=True)
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
        with open(filepath, "w") as f:
            f.write(
                self.model_dump_json(
                    indent=4, exclude_none=True, serialize_as_any=True
                )
            )

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

    def build_dag(self) -> dict[str, list[str]]:
        """Build directed acyclic graph (DAG) from the stages and their dependencies."""
        # TODO: Placeholder until DAG class is implemented, assuming we need one
        return self.dependencies

    def validate_dag(self):
        """Validate that the DAG structure is correct."""
        for stage, dependencies in self.dependencies.items():
            # Check for undefined dependencies
            for dep in dependencies:
                if dep not in self.stages:
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

    @classmethod
    def evaluate(
        cls, filepath: Path | str, method: str = "run", *args, **kwargs
    ) -> None:
        """Load pipeline and evaluate method.

        Parameters
        ----------
        filepath : Path or str
            Path to config file.
        method : str, optional
            Name of method to evaluate. Default is 'run'.

        """
        pipeline = cls.from_json(filepath)
        try:
            pipeline.__getattribute__(method)(*args, **kwargs)
        except AttributeError:
            raise AttributeError(
                f"{pipeline.name} does not have a '{method}' method"
            )

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


if __name__ == "__main__":
    fire.Fire(Pipeline.evaluate)
