"""Functions to run pipelines and stages with Jobmon.

Examples
--------
Compute resources can be passed as a dictionary or a path to a resources
file (e.g., json, toml, yaml).

Required tool resources:

.. code-block:: yaml

    tool_resources:
      {cluster_name}:
        project: proj_name
        queue: queue_name.q

Optional stage resources can be specified at the stage or stage + method
level:

    task_template_resources:
      {stage_name}:
        {cluster_name}:
            ...
      {stage_name}_{collect}:
        {cluster_name}:
            ...

See Jobmon documentation for additional optional resources and default
values.

"""

import sys
from pathlib import Path
from typing import Literal

from jobmon.client.api import Tool
from jobmon.client.task import Task
from jobmon.client.task_template import TaskTemplate
from pydantic import validate_call

from onemod.fsutils.config_loader import ConfigLoader
from onemod.pipeline import Pipeline
from onemod.stage import ModelStage, Stage


def get_tool(name: str, cluster: str, resources: dict) -> Tool:
    """Get tool."""
    tool = Tool(name=f"{name}")
    tool.set_default_cluster_name(cluster)
    tool.set_default_compute_resources_from_dict(
        cluster, resources["tool_resources"][cluster]
    )
    return tool


def run_workflow(name: str, tool: Tool, tasks: list[Task]) -> None:
    """Create and run workflow."""
    workflow = tool.create_workflow(name=f"{name}")
    workflow.add_tasks(tasks)
    workflow.bind()
    print(f"Starting workflow {workflow.workflow_id}")
    status = workflow.run()
    if status != "D":
        raise ValueError(f"Workflow {workflow.workflow_id} failed")
    else:
        print(f"Workflow {workflow.workflow_id} finished")


def get_tasks(
    tool: Tool,
    resources: dict,
    stage: Stage,
    method: str,
    task_args: dict[str, str],
    upstream_tasks: list[Task] = [],
) -> list[Task]:
    """Get stage tasks."""
    node_args = {}
    if isinstance(stage, ModelStage) and method != "collect":
        for node_arg in ["subset_id", "param_id"]:
            if len(node_vals := getattr(stage, node_arg + "s")) > 0:
                node_args[node_arg] = node_vals

    task_template = get_task_template(
        tool, resources, stage.name, method, list(node_args.keys())
    )

    if node_args:
        tasks = task_template.create_tasks(
            name=f"{stage.name}_{method}",
            upstream_tasks=upstream_tasks,
            max_attempts=1,
            **{**task_args, **node_args},
        )
    else:
        tasks = [
            task_template.create_task(
                name=f"{stage.name}_{method}",
                upstream_tasks=upstream_tasks,
                max_attempts=1,
                **task_args,
            )
        ]

    if isinstance(stage, ModelStage) and method in stage.collect_after:
        tasks.extend(
            get_tasks(tool, resources, stage, "collect", task_args, tasks)
        )

    return tasks


def get_task_template(
    tool: Tool,
    resources: dict,
    stage_name: str,
    method: str,
    node_args: list[str],
) -> TaskTemplate:
    """Get stage task template."""
    task_template = tool.get_task_template(
        template_name=f"{stage_name}_{method}",
        command_template=get_command_template(stage_name, method, node_args),
        node_args=node_args,
        task_args=["config"],
        op_args=["python"],
    )
    task_resources = get_task_resources(
        resources, tool.default_cluster_name, stage_name, method
    )
    if task_resources is not None:
        task_template.set_default_compute_resources_from_dict(
            tool.default_cluster_name, task_resources
        )
    return task_template


def get_command_template(
    stage_name: str, method: str, node_args: list[str]
) -> str:
    """Get stage command template."""
    command_template = (
        "{python}"
        f" {Path(__file__).parents[1] / 'main.py'}"
        " --config {config}"
        f" --method {method}"
        f" --stages {stage_name}"
    )

    for node_arg in node_args:
        command_template += f" --{node_arg} {{{node_arg}}}"

    return command_template


def get_task_resources(
    resources: dict, cluster: str, stage_name: str, method: str
) -> dict | None:
    """Get task-specific resources."""
    task_resources = resources.get("task_template_resources")
    if task_resources is not None:
        stage_resources = task_resources.get(stage_name, {})
        method_resources = task_resources.get(f"{stage_name}_{method}", {})
        if method_resources:
            return stage_resources.get(cluster)
        if stage_resources:
            return method_resources.get(cluster)
        return {
            **stage_resources.get(cluster, {}),
            **method_resources.get(cluster, {}),
        }
    return None


def get_upstream_tasks(
    stage: Stage,
    method: Literal["run", "fit", "predict"],
    stage_dict: dict[str, Stage],
    task_dict: dict[str, list[Task]],
    stages: set[str] | None = None,
) -> list[Task]:
    """Get upstream stage tasks.

    Parameters
    ----------
    stage : Stage
        Current stage.
    method : str
        Name of  method to evaluate.
    stage_dict : dict[str, Stage]
        Dictionary of all upstream pipeline stages.
    task_dict : dict[str, list[Task]]
        Dictionary of all tasks being evaluated.
    stages : set[str] or None, optional
        Name of all pipeline stages being evaluated.

    Returns
    -------
    list of Task
        Upstream tasks for current stage.

    """
    upstream_tasks = []

    for upstream_name in stage.dependencies:
        if stages is not None and upstream_name not in stages:
            continue

        upstream = stage_dict[upstream_name]
        if method not in upstream.skip:
            if (
                isinstance(upstream, ModelStage)
                and method in upstream.collect_after
            ):
                upstream_tasks.append(task_dict[upstream_name][-1])
            else:
                upstream_tasks.extend(task_dict[upstream_name])

    return upstream_tasks


@validate_call
def evaluate_with_jobmon(
    model: Pipeline | Stage,
    cluster: str,
    resources: Path | str | dict,
    python: Path | str | None = None,
    method: Literal["run", "fit", "predict"] = "run",
    stages: set[str] | None = None,
) -> None:
    """Evaluate pipeline or stage method with Jobmon.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance.
    cluster : str
        Cluster name.
    resources : Path, str, or dict
        Dictionary of compute resources or path to resources file.
    python : Path, str, or None, optional
        Path to Python environment. If None, use sys.executable.
        Default is None.
    method : str, optional
        Name of method to evalaute. Default is 'run'.
    stages : set of str or None, optional
        Names of stages to evaluate if `model` is a pipeline instance.
        If None, evaluate entire pipeline. Default is None.

    TODO: Optional stage-specific Python environments
    TODO: User-defined max_attempts
    TODO: Could dependencies be method specific?

    """
    # Get compute resources
    resources_dict: dict
    if isinstance(resources, (Path, str)):
        config_loader = ConfigLoader()
        resources_dict = config_loader.load(resources)
    else:
        resources_dict = resources

    # Get tool
    tool = get_tool(model.name, cluster, resources_dict)

    # Set config
    if isinstance(model, Stage):
        config_path = model.dataif.get_path("config")
    elif isinstance(model, Pipeline):
        config_path = model.directory / f"{model.name}.json"

    task_args: dict[str, str] = {
        "python": str(python or sys.executable),
        "config": str(config_path),
    }

    # Create tasks
    if isinstance(model, Pipeline):
        tasks = []
        task_dict: dict[str, list[Task]] = {}

        for stage_name in model.get_execution_order(stages):
            stage = model.stages[stage_name]
            if method not in stage.skip:
                upstream_tasks = get_upstream_tasks(
                    stage, method, model.stages, task_dict, stages
                )
                task_dict[stage_name] = get_tasks(
                    tool,
                    resources_dict,
                    stage,
                    method,
                    task_args,
                    upstream_tasks,
                )
                tasks.extend(task_dict[stage_name])
    else:
        tasks = get_tasks(tool, resources_dict, model, method, task_args)

    # Create and run workflow
    run_workflow(model.name, tool, tasks)
