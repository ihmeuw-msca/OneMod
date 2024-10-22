"""Functions to run pipelines and stages with Jobmon.

Examples
--------
Resources yaml file requires tool resources:

.. code-block:: yaml

    tool_resources:
      cluster_name:
        cores: 1
        memory: 1G
        project: proj_name
        queue: queue_name.q
        runtime: 00:01:00

Optional stage resources can be specified at the stage or method level:

    task_template_resources:
      stage_name:
        cluster_name:
          runtime: 00:10:00
      stage_name_collect:
        cluster_name:
          runtime: 00:05:00

In the above example, the stage's `collect` method requests five
minutes, while all other methods request ten minutes.

"""

import sys
from pathlib import Path
from typing import Literal

import yaml
from jobmon.client.api import Tool
from jobmon.client.task import Task
from jobmon.client.task_template import TaskTemplate
from pydantic import validate_call

from onemod.pipeline import Pipeline
from onemod.stage import ModelStage, Stage


def get_tool(name: str, cluster: str, resources: Path | str) -> Tool:
    """Get tool."""
    tool = Tool(name=f"{name}")
    tool.set_default_cluster_name(cluster)
    tool.set_default_compute_resources_from_yaml(cluster, resources)
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
    resources: Path | str,
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
        tool, resources, stage.name, method, node_args.keys()
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

    if isinstance(stage, ModelStage) and method != "collect":
        tasks.extend(
            get_tasks(tool, resources, stage, "collect", task_args, tasks)
        )

    return tasks


def get_task_template(
    tool: Tool,
    resources: Path | str,
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
        f" {Path(__file__).parents[1] / "main.py"}"
        " --config {config}"
        f" --stage_name {stage_name}"
        f" --method {method}"
    )

    for node_arg in node_args:
        command_template += f" --{node_arg} {{{node_arg}}}"

    return command_template


def get_task_resources(
    resources: Path | str, cluster: str, stage_name: str, method: str
) -> dict | None:
    """Get task-specific resources."""
    with open(resources, "r") as f:
        resource_dict = yaml.safe_load(f)["task_template_resources"]
    if f"{stage_name}_{method}" in resource_dict:
        return resource_dict[f"{stage_name}_{method}"][cluster]
    if stage_name in resource_dict:
        return resource_dict[stage_name][cluster]
    return None


@validate_call
def evaluate_with_jobmon(
    model: Pipeline | Stage,
    cluster: str,
    resources: Path | str,
    python: Path | str | None = None,
    method: Literal["run", "fit", "predict", "collect"] = "run",
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

    TODO: Optional stage-specific Python environments
    TODO: User-defined max_attempts

    """
    # Get tool
    tool = get_tool(model.name, cluster, resources)

    # Create tasks
    task_args = {
        "python": python or sys.executable,
        "config": str(model.dataif.config),
    }
    if isinstance(model, Pipeline):
        tasks = []
        task_dict = {}
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
                    tool, resources, stage, method, task_args, upstream_tasks
                )
                tasks.extend(task_dict[stage_name])
    else:
        tasks = get_tasks(tool, resources, model, method, task_args)

    # Create and run workflow
    run_workflow(model.name, tool, tasks)
