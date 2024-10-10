"""Jobmon functions to run pipelines and stages."""

import sys
from pathlib import Path
from typing import Literal

try:
    from jobmon.client.api import Tool
    from jobmon.client.task import Task
    from jobmon.client.task_template import TaskTemplate
except ImportError:
    pass

from onemod.pipeline import Pipeline
from onemod.stage import ModelStage, Stage


def get_tool(name: str, cluster: str, resources: Path | str) -> Tool:
    """Get tool."""
    tool = Tool(name=f"{name}_tool")
    tool.set_default_cluster_name(cluster)
    tool.set_default_compute_resources_from_yaml(cluster, resources)
    return tool


def run_workflow(name: str, tool: Tool, tasks: list[Task]) -> None:
    """Create and run workflow."""
    workflow = tool.create_workflow(name=f"{name}_workflow")
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
    stage: Stage,
    method: str,
    task_args: dict[str, str],
    upstream_tasks: list[Task] = [],
) -> list[Task]:
    """Get stage tasks."""
    node_args = {}
    if isinstance(stage, ModelStage) and method != "collect":
        for node_arg in ["subset_id", "param_id"]:
            if hasattr(stage, node_arg + "s"):
                if len(node_vals := getattr(stage, node_arg + "s")) > 0:
                    node_args[node_arg] = node_vals

    task_template = get_task_template(
        tool, stage.name, method, node_args.keys()
    )

    if node_args:
        tasks = task_template.create_tasks(
            name=f"{stage.name}_{method}_task",
            upstream_tasks=upstream_tasks,
            max_attempts=1,
            **{**task_args, **node_args},
        )
    else:
        tasks = [
            task_template.create_task(
                name=f"{stage.name}_{method}_task",
                upstream_tasks=upstream_tasks,
                max_attempts=1,
                **task_args,
            )
        ]

    if isinstance(stage, ModelStage) and method != "collect":
        tasks.extend(get_tasks(tool, stage, "collect", task_args, tasks))

    return tasks


def get_task_template(
    tool: Tool, stage_name: str, method: str, node_args: list[str]
) -> TaskTemplate:
    """Get stage task template."""
    return tool.get_task_template(
        template_name=f"{stage_name}_{method}_template",
        command_template=get_command_template(stage_name, method, node_args),
        node_args=node_args,
        task_args=["config", "from_pipeline"],
        op_args=["python"],
    )


def get_command_template(
    stage_name: str, method: str, node_args: list[str]
) -> str:
    """Get stage command template."""
    command_template = (
        "{python}"
        f" {Path(__file__).parents[1] / "main.py"}"
        " --config {config}"
        f" --stage_name {stage_name}"
        " --from_pipeline {from_pipeline}"
        f" --method {method}"
    )

    for node_arg in node_args:
        command_template += f" --{node_arg} {{{node_arg}}}"

    return command_template


def evaluate_with_jobmon(
    model: Pipeline | ModelStage,
    cluster: str,
    resources: Path | str,
    python: Path | str | None = None,
    method: Literal["run", "fit", "predict"] = "run",
    config: Path | str | None = None,
    from_pipeline: bool = False,
) -> None:
    """Evaluate pipeline or stage with Jobmon.

    Parameters
    -----------
    model : Pipeline or Stage
        Pipeline or Stage instance, specifically stages with either
        `groupby` or `crossby` attributes.
    cluster : str
        Cluster name.
    resources : Path or str
        Path to resources yaml file.
    python : Path, str or None, optional
        Path to python environment. If None, use sys.executable.
        Default is None.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    config : Path, str or None, optional
        Path to config file. Only used if `model` is a Stage instance.
        If None, model.directory / (model.name + ".json") used.
        Default is None.
    from_pipeline : bool, optional
        Whether `config` is a pipeline or stage config file. Only used
        if `model` is a Stage instance. Default is False.

    TODO: Optional stage-specific python environments
    TODO: User-defined max_attempts

    """
    # Get tool
    tool = get_tool(model.name, cluster, resources)

    # Create tasks
    if isinstance(model, Pipeline):
        tasks = []
        upstream_tasks = []
        task_args = {
            "python": python or sys.executable,
            "config": str(model.directory / (model.name + ".json")),
            "from_pipeline": True,
        }
        for stage in model.stages.values():
            if method not in stage.skip:
                upstream_tasks = get_tasks(
                    tool, stage, method, task_args, upstream_tasks
                )
                tasks.extend(upstream_tasks)
    else:
        task_args = {
            "python": sys.executable,
            "config": config or str(model.directory / (model.name + ".json")),
            "from_pipeline": from_pipeline,
        }
        tasks = get_tasks(tool, model, method, task_args)

    # Create and run workflow
    run_workflow(model.name, tool, tasks)
