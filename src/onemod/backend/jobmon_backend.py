"""Functions to run pipelines and stages with Jobmon."""

import sys
from pathlib import Path
from typing import Literal

from jobmon.client.api import Tool
from jobmon.client.task import Task
from jobmon.client.task_template import TaskTemplate
from pydantic import validate_call

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
        task_args=["config"],
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
        f" --method {method}"
    )

    for node_arg in node_args:
        command_template += f" --{node_arg} {{{node_arg}}}"

    return command_template


@validate_call
def evaluate_with_jobmon(
    model: Pipeline | Stage,
    cluster: str,
    resources: Path | str,
    python: Path | str | None = None,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    config: Path | str | None = None,
) -> None:
    """Evaluate pipeline or stage method with Jobmon.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance.
    cluster : str
        Cluster name.
    resources : Path or str
        Path to resources yaml file.
    python : Path, str, or None, optional
        Path to Python environment. If None, use sys.executable.
        Default is None.
    method : str, optional
        Name of method to evalaute. Default is 'run'.
    config : Path, str, or None, optional
        Path to config file. If None, use
        pipeline.directory / (pipeline.name + ".json") or
        stage.directory.parent / (stage.pipeline.name + ".json").
        Default is None.

    TODO: Optional stage-specific Python environments
    TODO: User-defined max_attempts

    """
    # Get tool
    tool = get_tool(model.name, cluster, resources)

    # Create tasks
    if isinstance(model, Pipeline):
        tasks = []
        task_dict = {}
        task_args = {
            "python": python or sys.executable,
            "config": config or str(model.directory / (model.name + ".json")),
        }
        for stage_name in model.get_execution_order():
            stage = model.stages[stage_name]
            if method not in stage.skip:
                upstream_tasks = []
                for dep in stage.dependencies:
                    if isinstance(model.stages[dep], ModelStage):
                        upstream_tasks.append(task_dict[dep][-1])  # collect
                    else:
                        upstream_tasks.extend(task_dict[dep])
                task_dict[stage_name] = get_tasks(
                    tool, stage, method, task_args, upstream_tasks
                )
                tasks.extend(task_dict[stage_name])
    else:
        task_args = {
            "python": python or sys.executable,
            "config": config
            or str(model.directory.parent / (model.pipeline + ".json")),
        }
        tasks = get_tasks(tool, model, method, task_args)

    # if isinstance(model, Pipeline):
    #     tasks = []
    #     upstream_tasks = []
    #     task_args = {
    #         "python": python or sys.executable,
    #         "config": config or str(model.directory / (model.name + ".json")),
    #     }
    #     for stage_name in model.get_execution_order():
    #         stage = model.stages[stage_name]
    #         if method not in stage.skip:
    #             upstream_tasks = get_tasks(
    #                 tool, stage, method, task_args, upstream_tasks
    #             )
    #             tasks.extend(upstream_tasks)
    # else:
    #     task_args = {
    #         "python": python or sys.executable,
    #         "config": config
    #         or str(model.directory.parent / (model.pipeline + ".json")),
    #     }
    #     tasks = get_tasks(tool, model, method, task_args)

    # Create and run workflow
    run_workflow(model.name, tool, tasks)
