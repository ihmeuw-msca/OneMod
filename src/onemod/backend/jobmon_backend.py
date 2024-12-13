"""Functions to run pipelines and stages with Jobmon.

Examples
--------
Compute resources can be passed as a dictionary or a path to a resources
file (e.g., json, toml, yaml).

Required tool resources:

.. code-block:: yaml

    tool_resources:
      {cluster_name}:
        project: {proj_name}
        queue: {queue_name}

To test workflow on a dummy cluster (i.e., run workflow without running
tasks), use:

.. code-block:: yaml

    tool_resources:
      dummy:
        queue: null.q

Optional stage resources can be specified at the stage or stage + method
level:

    task_template_resources:
      {stage_name}:
        {cluster_name}:
            ...
      {stage_name}_{collect}:
        {cluster_name}:
            ...

See Jobmon documentation for additional resources and default values.

"""

import sys
from json import tool
from pathlib import Path
from typing import Any, Literal

from jobmon.client.api import Tool
from jobmon.client.task import Task
from jobmon.client.task_template import TaskTemplate
from pydantic import validate_call

from onemod.fsutils.config_loader import ConfigLoader
from onemod.pipeline import Pipeline
from onemod.stage import ModelStage, Stage


@validate_call
def evaluate_with_jobmon(
    model: Pipeline | Stage,
    cluster: str,
    resources: Path | str | dict[str, Any],
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
    resources_dict = get_resources(resources)
    tool = get_tool(model.name, method, cluster, resources_dict)
    tasks = get_tasks(tool, resources_dict, model, method, python, stages)
    run_workflow(model.name, method, tool, tasks)


def get_resources(resources: Path | str | dict[str, Any]) -> dict[str, Any]:
    """Get dictionary of compute resources.

    Parameters
    ----------
    resources : Path, str, or dict
        Dictionary of compute resources or path to resources file.

    Returns
    -------
    dict
        Dictionary of compute resources.

    """
    if isinstance(resources, (Path, str)):
        config_loader = ConfigLoader()
        return config_loader.load(Path(resources))
    return resources


def get_tool(
    name: str, method: str, cluster: str, resources: dict[str, Any]
) -> Tool:
    """Get Jobmon tool.

    Parameters
    ----------
    name : str
        Pipeline or stage name.
    method : str
        Name of method to evaluate.
    cluster : str
        Cluster name.
    resources : dict
        Dictionary of compute resources.

    Returns
    -------
    Tool
        Jobmon tool.

    """
    tool = Tool(name=f"{name}_{method}")
    tool.set_default_cluster_name(cluster)
    tool.set_default_compute_resources_from_dict(
        cluster, resources["tool_resources"][cluster]
    )
    return tool


def get_tasks(
    tool: tool,
    resources: dict[str, Any],
    model: Pipeline | Stage,
    method: str,
    python: Path | str | None,
    stages: list[str] | None,
) -> list[Task]:
    """Get Jobmon tasks.

    Parameters
    ----------
    tool : Tool
        Jobmon tool.
    resources : dict
        Dictionary of compute resources.
    model : Pipeline or Stage
        Pipeline or stage instance.
    method : str
        Name of method to evaluate.
    python : str or None
        Path to Python environment. If None, use sys.executable.
    stages : list of str or None
        Name of stages to evaluate if `model` is a pipeline instance. If
        None, evaluate entire pipeline.

    Returns
    -------
    list of Task
        List of Jobmon tasks.

    """
    task_args = (
        {"python": str(python or sys.executable), "config": get_config(model)},
    )
    if isinstance(model, Pipeline):
        return get_pipeline_tasks(
            tool, resources, model, method, task_args, stages
        )
    return get_stage_tasks(tool, resources, model, method, task_args)


def get_config(model: Pipeline | Stage) -> str:
    """Get path to pipeline config file.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance.

    Returns
    -------
    str
        Path to pipeline config file.

    """
    if isinstance(model, Pipeline):
        return str(model.directory / f"{model.name}.json")
    return str(model.dataif.get_path("config"))


def get_pipeline_tasks(
    tool: Tool,
    resources: dict[str, Any],
    pipeline: Pipeline,
    method: str,
    task_args: dict[str, str],
    stages: list[str] | None,
) -> list[Task]:
    """Get pipeline stage tasks.

    Parameters
    ----------
    tool : Tool
        Jobmon tool.
    resources : dict
        Dictionary of compute resources.
    pipeline : Pipeline
        Pipeline instance.
    method : str
        Name of method to evaluate.
    task_args : dict
        Dictionary containing paths to python environment and pipeline
        config file.
    stages : list of str or None
        Name of stages to evaluate if `model` is a pipeline instance. If
        None, evaluate entire pipeline.

    Returns
    -------
    list of Task
        List of pipeline stage tasks.

    """
    tasks = []
    task_dict: dict[str, list[Task]] = {}

    for stage_name in pipeline.get_execution_order(stages):
        stage = pipeline.stages[stage_name]
        if method not in stage.skip:
            upstream_tasks = get_upstream_tasks(
                stage, method, pipeline.stages, task_dict, stages
            )
            task_dict[stage_name] = get_stage_tasks(
                tool, resources, stage, method, task_args, upstream_tasks
            )
            tasks.extend(task_dict[stage_name])

    return tasks


def get_upstream_tasks(
    stage: Stage,
    method: Literal["run", "fit", "predict"],
    stage_dict: dict[str, Stage],
    task_dict: dict[str, list[Task]],
    stages: set[str] | None = None,
) -> list[Task]:
    """Get upstream tasks for current stage.

    Only include tasks corresponding to the current stage's
    dependencies. If a dependency is an instance of `ModelStage`, only
    include the task corresponding to the model stage's 'collect'
    method.

    Parameters
    ----------
    stage : Stage
        Current stage instance.
    method : str
        Name of method to evaluate.
    stage_dict : dict
        Dictionary of all pipeline stages.
    task_dict : dict
        Dictionary of all upstream stage tasks.
    stages : set of str or None, optional
        Names of all pipeline stages being evaluated. If None, assume
        all stages are being evaluated.

    Returns
    -------
    list of Task
        Upstream stage tasks for current stage.

    """
    upstream_tasks = []

    for upstream_name in stage.dependencies:
        if stages is not None and upstream_name not in stages:
            # upstream stage not being evaluated
            continue

        upstream_stage = stage_dict[upstream_name]
        if method not in upstream_stage.skip:
            if (
                isinstance(upstream_stage, ModelStage)
                and method in upstream_stage.collect_after
            ):
                # only include task corresponding to 'collect' method
                upstream_tasks.append(task_dict[upstream_name][-1])
            else:
                upstream_tasks.extend(task_dict[upstream_name])

    return upstream_tasks


def get_stage_tasks(
    tool: Tool,
    resources: dict[str, Any],
    stage: Stage,
    method: str,
    task_args: dict[str, str],
    upstream_tasks: list[Task] = [],
) -> list[Task]:
    """Get stage tasks.

    If stage is an instance of `ModelStage` and method is not 'collect',
    get tasks for all subset_id and param_id values and task for collect
    method.

    Parameters
    ----------
    tool : Tool
        Jobmon tool.
    resources : dict
        Dictionary of compute resources.
    stage : Stage
        Stage instance.
    method : str
        Name of method to evaluate.
    task_args : dict
        Dictionary containing paths to python environment and pipeline
        config file.
    upstream_tasks : list of Task, optional
        List of upstream stage tasks. Default is an empty list.

    Returns
    -------
    list of Task
        List of stage tasks.

    """
    node_args = {}
    if isinstance(stage, ModelStage) and method != "collect":
        # get all subset_id and/or param_id values
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
        # get task for collect method
        tasks.extend(
            get_stage_tasks(tool, resources, stage, "collect", task_args, tasks)
        )

    return tasks


def get_task_template(
    tool: Tool,
    resources: dict[str, Any],
    stage_name: str,
    method: str,
    node_args: list[str],
) -> TaskTemplate:
    """Get stage task template.

    Parameters
    ----------
    tool : Tool
        Jobmon tool.
    resources : dict
        Dictionary of compute resources.
    stage_name : str
        Stage name.
    method : str
        Name of method being evaluated.
    node_args : list of str
        List including 'subset_id' and/or 'param_id' if stage is an
        instance of `ModelStage` and `method` is not 'collect'.

    Returns
    -------
    TaskTemplate
        Stage task template.

    """
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
    if task_resources:
        task_template.set_default_compute_resources_from_dict(
            tool.default_cluster_name, task_resources
        )

    return task_template


def get_command_template(
    stage_name: str, method: str, node_args: list[str]
) -> str:
    """Get stage command template.

    All stages methods are called via `onemod.main.evaluate()`. If stage
    is an instance of `ModelStage` and `method` is not 'collect',
    additional args for 'subset_id' and/or 'param_id' are included in
    the command template.

    Parameters
    ----------
    stage_name : str
        Stage name.
    method : str
        Name of method being evaluated.
    node_args : list of str
        List including 'subset_id' and/or 'param_id' if stage is an
        instance of `ModelStage` and `method` is not 'collect'.

    Returns
    -------
    str
        Stage command template.

    """
    command_template = (
        "{python}"
        f" {Path(__file__).parents[1] / 'main.py'}"
        " --config {config}"
        f" --method {method}"
        f" --stages {stage_name}"
    )

    for node_arg in node_args:
        # add 'subset_id' and/or 'param_id'
        command_template += f" --{node_arg} {{{node_arg}}}"

    return command_template


def get_task_resources(
    resources: dict[str, Any], cluster: str, stage_name: str, method: str
) -> dict[str, Any]:
    """Get task-specific resources.

    Parameters
    ----------
    resources : dict
        Dictionary of compute resources.
    cluster : str
        Cluster name.
    stage_name : str
        Stage name.
    method : str
        Name of method being evaluated.

    Returns
    -------
    dict
        Task-specific resources.

    """
    task_resources = resources.get("task_template_resources", {})
    stage_resources = task_resources.get(stage_name, {})
    method_resources = task_resources.get(f"{stage_name}_{method}", {})
    return {
        **stage_resources.get(cluster, {}),
        **method_resources.get(cluster, {}),
    }


def run_workflow(name: str, method: str, tool: Tool, tasks: list[Task]) -> None:
    """Create and run workflow.

    Parameters
    ----------
    name : str
        Pipeline or stage name.
    method : str
        Name of method being evaluated.
    tool : Tool
        Jobmon tool.
    tasks : list of Task
        List of stage tasks.

    """
    workflow = tool.create_workflow(name=f"{name}_{method}")
    workflow.add_tasks(tasks)
    workflow.bind()
    print(f"Starting workflow {workflow.workflow_id}")
    status = workflow.run()
    if status != "D":
        raise ValueError(f"Workflow {workflow.workflow_id} failed")
    else:
        print(f"Workflow {workflow.workflow_id} finished")
