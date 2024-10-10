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
    stage: "Stage",
    method: str,
    task_args: dict[str, str],
    upstream_tasks: list[Task] = [],
) -> list[Task]:
    """Get stage tasks."""
    node_args = {}
    for id_name in ["subset_id", "param_id"]:
        if hasattr(stage, id_name + "s"):
            if len(id_vals := getattr(stage, id_name + "s")) > 0:
                node_args[id_name] = id_vals

    task_template = get_task_template(
        tool=tool,
        stage_name=stage.name,
        method=method,
        subsets="subset_id" in node_args,
        params="param_id" in node_args,
    )

    if node_args and method != "collect":
        tasks = task_template.create_tasks(
            name=f"{stage.name}_{method}_task",
            upstream_tasks=upstream_tasks,
            max_attempts=1,
            **{**task_args, **node_args},
        )
        tasks.extend(get_tasks(tool, stage, "collect", task_args, tasks))
    else:
        tasks = [
            task_template.create_task(
                name=f"{stage.name}_{method}_task",
                upstream_tasks=upstream_tasks,
                max_attempts=1,
                **task_args,
            )
        ]

    return tasks


def get_task_template(
    tool: Tool,
    stage_name: str,
    method: str,
    subsets: bool = False,
    params: bool = False,
) -> TaskTemplate:
    """Get stage task template."""
    node_args = []
    if subsets and method != "collect":
        node_args.append("subset_id")
    if params and method != "collect":
        node_args.append("param_id")

    return tool.get_task_template(
        template_name=f"{stage_name}_{method}_template",
        command_template=get_command_template(
            stage_name, method, subsets, params
        ),
        node_args=node_args,
        task_args=["config"],
        op_args=["python"],
    )


def get_command_template(
    stage_name: str, method: str, subsets: bool = False, params: bool = False
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

    if subsets and method != "collect":
        command_template += " --subset_id {subset_id}"
    if params and method != "collect":
        command_template += " --param_id {param_id}"

    return command_template


def evaluate_pipeline_with_jobmon(
    pipeline: "Pipeline",
    cluster: str,
    resources: Path | str,
    method: Literal["run", "fit", "predict"] = "run",
) -> None:
    """Evaluate pipeline with Jobmon.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline instance.
    cluster : str
        Cluster name.
    resources : Path or str
        Path to resources yaml file.
    method : str, optional
        Name of method to evaluate. Default is 'run'.

    # TODO: Optional stage-specific resources
    # TODO: Optional stage-specific python environments

    """
    # Get tool
    tool = get_tool(pipeline.name, cluster, resources)

    # Create tasks
    tasks = []
    upstream_tasks = []
    task_args = {
        "python": sys.executable,
        "config": str(pipeline.directory / (pipeline.name + ".json")),
    }
    for stage in pipeline.stages.values():
        if method not in stage.skip_if:
            upstream_tasks = get_tasks(
                tool, stage, method, task_args, upstream_tasks
            )
            tasks.extend(upstream_tasks)

    # Create and run workflow
    run_workflow(pipeline.name, tool, tasks)


def evaluate_stage_with_jobmon(
    stage: "GroupedStage" | "CrossedStage" | "ModelStage",
    config: Path | str,
    from_pipeline: bool,
    cluster: str,
    resources: Path | str,
    method: Literal["run", "fit", "predict"] = "run",
) -> None:
    """Evaluate stage with Jobmon.

    stage : Stage
        Stage instance.
    config : Path or str
        Path to config file.
    from_pipeline : bool
        Whether `config` is a pipeline or stage config file.
    cluster : str
        Cluster name.
    resources : Path or str
        Path to resources yaml file.
    method : str, optional
        Name of method to evaluate. Default is 'run'.

    # TODO: See if this runs
    # TODO: Make config and from_pipeline optional?
    # TODO: Combine with evaluate_pipeline_from_jobmon?

    """
    if method in stage.skip_if:
        raise AttributeError(f"{stage.name} skips the '{method}' method")
    if not hasattr(stage, method):
        raise AttributeError(f"{stage.name} does not have a '{method}' method")

    # Create tool
    tool = get_tool(stage.name, cluster, resources)

    # Create tasks
    if config is None:
        config = str(stage.directory / (stage.name + "json"))
    task_args = {"python": sys.executable, "config": config}
    tasks = get_tasks(tool, stage, method, task_args)

    # Create and run workflow
    run_workflow(stage.name, tool, tasks)
