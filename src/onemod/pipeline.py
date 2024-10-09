"""Pipeline class."""

from __future__ import annotations

import fire
import json
import logging
from collections import deque
from itertools import product
from pathlib import Path
import sys

from jobmon.client.api import Tool
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

    def evaluate(
        self, method: str = "run", use_jobmon: bool = False, *args, **kwargs
    ) -> None:
        """Evaluate pipeline method.

        Parameters
        ----------
        method : str, optional
            Name of method to evaluate. Default is 'run'.

        TODO: Add options to run subset of stages
        TODO: Add options to run subset of IDs

        """
        if use_jobmon:
            self.evaluate_jobmon(method=method, *args, **kwargs)
        else:
            self.evaluate_local(method=method, *args, **kwargs)

    def evaluate_local(self, method: str = "run", *args, **kwargs) -> None:
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

    def evaluate_jobmon(
        self,
        cluster_name: str,
        resources_yaml: str,
        method: str = "run",
        *args,
        **kwargs,
    ) -> None:
        # Create tool
        tool = Tool(name="example_tool")
        tool.set_default_cluster_name(cluster_name)
        tool.set_default_compute_resources_from_yaml(
            cluster_name, resources_yaml, set_task_templates=True
        )

        # Create tasks
        tasks = []
        upstream_tasks = []
        task_args = {
            "python": sys.executable,
            "filepath": self.directory / (self.name + ".json"),
        }
        for stage in self.stages.values():
            if method not in stage._skip_if:
                node_args = {}
                if isinstance(stage, GroupedStage) and stage.subset_ids:
                    node_args["subset_id"] = stage.subset_ids
                if isinstance(stage, CrossedStage) and stage.param_ids:
                    node_args["param_id"] = stage.param_ids

                task_template = onemod.get_task_template(
                    tool=tool,
                    stage_name=stage.name,
                    method=method,
                    subsets="subset_id" in node_args,
                    params="param_id" in node_args,
                )

                if "subset_id" in node_args or "param_id" in node_args:
                    upstream_tasks = task_template.create_tasks(
                        name=f"{stage.name}_{method}_task",
                        upstream_tasks=upstream_tasks,
                        max_attempts=1,
                        **{**task_args, **node_args},
                    )
                    tasks.extend(upstream_tasks)

                    upstream_tasks = [
                        onemod.get_task_template(
                            tool=tool, stage_name=stage.name, method="collect"
                        ).create_task(
                            name=f"{stage.name}_collect_task",
                            upstream_tasks=upstream_tasks,
                            max_attempts=1,
                            **task_args,
                        )
                    ]
                    tasks.extend(upstream_tasks)
                else:
                    upstream_tasks = [
                        task_template.create_task(
                            name=f"{stage.name}_{method}_task",
                            upstream_tasks=upstream_tasks,
                            max_attempts=1,
                            **task_args,
                        )
                    ]
                    tasks.extend(upstream_tasks)

        # Create and run workflow
        workflow = tool.create_workflow(name="example_workflow")
        workflow.add_tasks(tasks)
        workflow.bind()
        print(f"workflow_id: {workflow.workflow_id}")
        status = workflow.run()
        if status != "D":
            raise ValueError(f"Workflow {workflow.workflow_id} failed")
        else:
            print(f"Workflow {workflow.workflow_id} finished")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


if __name__ == "__main__":
    fire.Fire(Pipeline.evaluate)
