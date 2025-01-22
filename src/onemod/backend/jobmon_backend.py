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
from pathlib import Path
from typing import Any, Literal

from jobmon.client.api import Tool
from jobmon.client.task import Task
from jobmon.client.task_template import TaskTemplate
from pydantic import validate_call

from onemod.backend.utils import check_input, check_method
from onemod.dtypes import UniqueList
from onemod.fsutils.config_loader import ConfigLoader
from onemod.pipeline import Pipeline
from onemod.stage import Stage


@validate_call
def evaluate(
    model: Pipeline | Stage,
    cluster: str,
    resources: Path | str | dict[str, Any],
    python: Path | str | None = None,
    method: Literal["run", "fit", "predict"] = "run",
    stages: UniqueList[str] | None = None,
    **kwargs,
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
    stages : list of str or None, optional
        Names of stages to evaluate if `model` is a `Pipeline` instance.
        If None, evaluate entire pipeline. Default is None.

    Other Parameters
    ----------------
    **kwargs
        Additional keyword arguments passed to stage methods.

    """
    # TODO: Optional stage-specific Python environments
    # TODO: User-defined max_attempts
    # TODO: Could dependencies be method specific?
    check_method(model, method, backend="jobmon")
    check_input(model, stages)
    resources_dict = get_resources(resources)
    tool = get_tool(model.name, method, cluster, resources_dict)
    tasks = get_tasks(
        tool, resources_dict, model, method, python, stages, **kwargs
    )
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
    # TODO: should we check format, minimum resources, cluster?
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
    tool: Tool,
    resources: dict[str, Any],
    model: Pipeline | Stage,
    method: Literal["run", "fit", "predict"],
    python: Path | str | None,
    stages: list[str] | None,
    **kwargs,
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
    python : Path, str, or None
        Path to Python environment. If None, use sys.executable.
    stages : list of str or None
        Name of stages to evaluate if `model` is a pipeline instance. If
        None, evaluate entire pipeline.

    Other Parameters
    ----------------
    **kwargs
        Additional keyword arguments passed to stage methods.

    Returns
    -------
    list of Task
        List of Jobmon tasks.

    """
    if isinstance(model, Pipeline):
        return get_pipeline_tasks(
            tool, resources, model, method, python, stages, **kwargs
        )
    return get_stage_tasks(tool, resources, model, method, python, **kwargs)


def get_pipeline_tasks(
    tool: Tool,
    resources: dict[str, Any],
    pipeline: Pipeline,
    method: Literal["run", "fit", "predict"],
    python: Path | str | None,
    stages: list[str] | None,
    **kwargs,
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
    python : Path, str, or None
        Path to Python environment. If None, use sys.executable.
    stages : list of str or None
        Name of stages to evaluate if `model` is a pipeline instance. If
        None, evaluate entire pipeline.

    Other Parameters
    ----------------
    **kwargs
        Additional keyword arguments passed to stage methods.

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
                tool, resources, stage, method, python, upstream_tasks, **kwargs
            )
            tasks.extend(task_dict[stage_name])

    return tasks


def get_upstream_tasks(
    stage: Stage,
    method: Literal["run", "fit", "predict"],
    stage_dict: dict[str, Stage],
    task_dict: dict[str, list[Task]],
    stages: list[str] | None = None,
) -> list[Task]:
    """Get upstream tasks for current stage.

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
    stages : list of str or None, optional
        Names of all pipeline stages being evaluated. If None, assume
        all stages are being evaluated.

    Returns
    -------
    list of Task
        Upstream stage tasks for current stage.

    Notes
    -----
    * Only include tasks corresponding to the current stage's
      dependencies that are included in `stages`.
    * If an upstream stage has submodels and `method` is in the
      upstream's `collect_after`, only include the task corresponding to
      the upstream's `collect` method.

    """
    upstream_tasks = []

    for upstream_name in stage.dependencies:
        if stages is not None and upstream_name not in stages:
            # upstream stage not being evaluated
            continue

        upstream_stage = stage_dict[upstream_name]
        if method not in upstream_stage.skip:
            if (
                upstream_stage.has_submodels
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
    method: Literal["run", "fit", "predict", "collect"],
    python: Path | str | None,
    upstream_tasks: list[Task] | None = None,
    **kwargs,
) -> list[Task]:
    """Get stage tasks.

    If stage has submodels and `method` is not 'collect', get tasks for
    all submodels. If `method` not in stage's `collect_after`, add a
    task for stage's `collect` method.

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
    python : Path, str, or None
        Path to Python environment. If None, use sys.executable.
    upstream_tasks : list of Task or None, optional
        List of upstream stage tasks. Default is None.

    Other Parameters
    ----------------
    **kwargs
        Additional keyword arguments passed to stage methods.

    Returns
    -------
    list of Task
        List of stage tasks.

    """
    if upstream_tasks is None:
        upstream_tasks = []

    entrypoint = get_entrypoint(python)
    config_path = get_config_path(stage)
    node_args = get_node_args(stage, method)

    task_template = get_task_template(
        tool,
        resources,
        stage.name,
        method,
        list(node_args.keys()) + list(kwargs.keys()),
    )

    if node_args:
        tasks = task_template.create_tasks(
            name=f"{stage.name}_{method}",
            upstream_tasks=upstream_tasks,
            max_attempts=1,
            entrypoint=entrypoint,
            config=config_path,
            **node_args,
            **kwargs,
        )
    else:
        tasks = [
            task_template.create_task(
                name=f"{stage.name}_{method}",
                upstream_tasks=upstream_tasks,
                max_attempts=1,
                entrypoint=entrypoint,
                config=config_path,
                **kwargs,
            )
        ]

    if stage.has_submodels and method in stage.collect_after:
        # get task for collect method
        tasks.extend(
            get_stage_tasks(
                tool, resources, stage, "collect", entrypoint, tasks
            )
        )

    return tasks


def get_entrypoint(python: Path | str | None = None) -> str:
    """Get path to python entrypoint.

    All stages methods are called via `onemod.main.evaluate()`.

    Parameters
    ----------
    python : Path or str, optional
        Path to python environment. If None, use sys.executable.
        Default is None.

    Returns
    -------
    str
        Path to python entrypoint.

    """
    return str(Path(python or sys.executable).parent / "onemod")


def get_config_path(stage: Stage) -> str:
    """Get path to pipeline config file.

    Parameters
    ----------
    stage : Stage
        Stage instance.

    Returns
    -------
    str
        Path to pipeline config file.

    """
    return str(stage.dataif.get_path("config"))


def get_node_args(
    stage: Stage, method: Literal["run", "fit", "predict", "collect"]
) -> dict[str, Any]:
    """Get dictionary of subset and/or paramset values.

    If stage has submodels and `method` is not 'collect', additional
    args for 'subset' and/or 'paramset' are included in the command
    template.

    Parameters
    ----------
    stage : Stage
        Stage instance.
    method : str
        Method being evaluated.

    Returns
    -------
    dict
        Dictionary of subset and/or paramset values.

    """
    node_args = {}
    if stage.has_submodels and method != "collect":
        for attr, node_arg in [
            ["subsets", "subset"],
            ["paramsets", "paramset"],
        ]:
            if (node_vals := getattr(stage, attr)) is not None:
                node_args[node_arg] = [
                    str(node_val)
                    for node_val in node_vals.to_dict(orient="records")
                ]
    return node_args


def get_task_template(
    tool: Tool,
    resources: dict[str, Any],
    stage_name: str,
    method: Literal["run", "fit", "predict", "collect"],
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
        List including 'subset', 'paramset', and/or other keyword
        arguments passed to stage methods.

    Returns
    -------
    TaskTemplate
        Stage task template.

    """
    task_template = tool.get_task_template(
        template_name=f"{stage_name}_{method}",
        command_template=get_command_template(stage_name, method, node_args),
        op_args=["entrypoint"],
        task_args=["config"],
        node_args=node_args,
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
    stage_name: str,
    method: Literal["run", "fit", "predict", "collect"],
    node_args: list[str],
) -> str:
    """Get stage command template.

    All stages methods are called via `onemod.main.evaluate()`. If stage
    has submodels and `method` is not 'collect', additional args for
    'subset' and/or 'paramset' are included in the command template.

    Parameters
    ----------
    stage_name : str
        Stage name.
    method : str
        Name of method being evaluated.
    node_args : list of str
        List including 'subset', 'paramset', and/or other keyword
        arguments passed to stage methods.

    Returns
    -------
    str
        Stage command template.

    """
    command_template = (
        "{entrypoint} --config {config}"
        f" --method {method} --stages {stage_name}"
    )

    for node_arg in node_args:
        # add 'subset', 'paramset' and/or other kwargs
        command_template += f" --{node_arg} {{{node_arg}}}"

    return command_template


def get_task_resources(
    resources: dict[str, Any],
    cluster: str,
    stage_name: str,
    method: Literal["run", "fit", "predict", "collect"],
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
