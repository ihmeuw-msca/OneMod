"""Jobmon functions to run pipelines and stages."""

import sys
from typing import Literal

from jobmon.client.api import Tool
from jobmon.client.task import Task
from jobmon.client.task_template import TaskTemplate

from onemod.stage import Stage, GroupedStage, CrossedStage


def get_tasks(
    tool: Tool,
    stage: Stage,
    method: str,
    upstream_tasks: list[Task],
    task_args: dict[str, str],
) -> list[Task]:
    """Get stage tasks."""
    node_args = {}
    if isinstance(stage, GroupedStage) and stage.subset_ids:
        node_args["subset_id"] = stage.subset_ids
    if isinstance(stage, CrossedStage) and stage.param_ids:
        node_args["param_id"] = stage.param_ids

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
        tasks.extend(get_tasks(tool, stage, "collect", tasks, task_args))
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
        task_args=["filepath"],
        op_args=["python"],
    )


def get_command_template(
    stage_name: str, method: str, subsets: bool = False, params: bool = False
) -> str:
    """Get stage command template."""
    command_template = (
        "{python}"
        f" {__file__}"
        " --filepath {filepath}"
        f" --stage_name {stage_name}"
        f" --method {method}"
        " --from_pipeline"
    )

    if subsets and method != "collect":
        command_template += " --subset_id {subset_id}"
    if params and method != "collect":
        command_template += " --param_id {param_id}"

    return command_template


def evaluate_with_jobmon(
    pipeline: "Pipeline",
    cluster: str,
    resources: str,
    method: Literal["run", "fit", "predict"] = "run",
    *args,
    **kwargs,
) -> None:
    """Run pipeline with Jobmon.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline instance.
    cluster : str
        Cluster name.
    resources : str
        Path to resources yaml file.
    method : str, optional
        Name of method to evaluate. Default is 'run'.

    # TODO: Run through to make sure still working
    # TODO: Optional stage-specific resources
    # TODO: Optional stage-specific python environments

    """
    # Create tool
    tool = Tool(name="onemod_tool")
    tool.set_default_cluster_name(cluster)
    tool.set_default_compute_resources_from_yaml(
        cluster, resources, set_task_templates=True
    )

    # Create tasks
    tasks = []
    upstream_tasks = []
    task_args = {
        "python": sys.executable,
        "filepath": str(pipeline.directory / (pipeline.name + ".json")),
    }
    for stage in pipeline.stages.values():
        if method not in stage._skip_if:
            upstream_tasks = get_tasks(
                tool, stage, method, upstream_tasks, task_args
            )
            tasks.extend(upstream_tasks)

    # Create and run workflow
    workflow = tool.create_workflow(name="onemod_workflow")
    workflow.add_tasks(tasks)
    workflow.bind()
    print(f"Starting workflow {workflow.workflow_id}")
    status = workflow.run()
    if status != "D":
        raise ValueError(f"Workflow {workflow.workflow_id} failed")
    else:
        print(f"Workflow {workflow.workflow_id} finished")
