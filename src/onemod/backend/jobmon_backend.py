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

from onemod.backend.utils import check_input_exists, check_method
from onemod.fsutils.config_loader import ConfigLoader
from onemod.pipeline import Pipeline
from onemod.stage import Stage


def evaluate(
    model: Pipeline | Stage,
    method: Literal["run", "fit", "predict", "collect"],
    cluster: str,
    resources: dict[str, Any] | Path | str,
    python: Path | str | None = None,
    method_args: dict[str, Any | dict[str, Any]] | None = None,
    stages: list[str] | None = None,
    subsets: dict[str, Any | list[Any]] | None = None,
    paramsets: dict[str, Any | list[Any]] | None = None,
    collect: bool = False,
) -> None:
    """Evaluate pipeline or stage method with Jobmon.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance.
    method : {'run', 'fit', 'predict', 'collect'}
        Name of method to evalaute.
    cluster : str
        Cluster name.
    resources : dict, Path, or str
        Path to resources file or dictionary of compute resources.
    python : Path or str, optional
        Path to Python environment. If None, use sys.executable.
        Default is None.
    method_args : dict, optional
        Additional keyword arguments passed to stage methods. If `model`
        is a `Pipeline` instance, use format
        `{stage_name: {arg_name: arg_value}}`. If `model` is a `Stage`
        instance, use format `{arg_name: arg_value}`.
    stages : list of str, optional
        Names of stages to evaluate if `model` is a `Pipeline` instance.
        If None, evaluate pipeline stages. Default is None.
    subsets : dict, optional
        Submodel data subsets to evaluate if `model` is a `Stage`
        instance. If None, evaluate all data subsets. Default is None.
    paramsets : dict, optional
        Submodel parameter sets to evaluate if `model` is a `Stage`
        instance. If None, evaluate all parameter sets. Default is None.
    collect : bool, optional
        Collect submodel results if `model` is a `Stage` instance and
        `subsets` and `params` are not both None. Default is False.
        If `subsets` and `paramsets` are both None, this parameter is
        ignored and submodel results are always collected.

    """
    # TODO: Optional stage-specific Python environments
    # TODO: User-defined max_attempts
    # TODO: Could dependencies be method specific?
    check_method(model, method)
    check_input_exists(model, stages)
    if python is None:
        python = str(sys.executable)
    if method_args is None:
        method_args = {}

    resources_dict = get_resources(resources)
    tool = get_tool(model.name, method, cluster, resources_dict)
    tasks = get_tasks(
        model,
        method,
        tool,
        resources_dict,
        python,
        method_args,
        stages,
        subsets,
        paramsets,
        collect,
    )
    run_workflow(model.name, method, tool, tasks)


def get_resources(resources: Path | str | dict[str, Any]) -> dict[str, Any]:
    """Get dictionary of compute resources.

    Parameters
    ----------
    resources : Path, str, or dict
        Path to resources file or dictionary of compute resources.

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
    model: Pipeline | Stage,
    method: str,
    tool: Tool,
    resources: dict[str, Any],
    python: Path | str,
    method_args: dict[str, Any | dict[str, Any]],
    stages: list[str] | None,
    subsets: dict[str, Any | list[Any]] | None,
    paramsets: dict[str, Any | list[Any]] | None,
    collect: bool,
) -> list[Task]:
    """Get Jobmon tasks.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance.
    method : str
        Name of method to evaluate.
    tool : Tool
        Jobmon tool.
    resources : dict
        Dictionary of compute resources.
    python : Path or str
        Path to Python environment.
    method_args : dict
        Additional keyword arguments passed to stage methods.
    stages : list of str or None
        Name of stages to evaluate if `model` is a pipeline instance. If
        None, evaluate all pipeline stages.
    subsets : dict
        Submodel data subsets to evaluate if `model` is a `Stage`
        instance. If None, evaluate all data subsets.
    paramsets : dict
        Submodel parameter sets to evaluate if `model` is a `Stage`
        instance. If None, evaluate all parameter sets.
    collect : bool
        Collect submodel results if `model` is a `Stage` instance and
        `subsets` and `params` are not both None.

    Returns
    -------
    list of Task
        List of Jobmon tasks.

    """
    if isinstance(model, Pipeline):
        return get_pipeline_tasks(
            model, method, tool, resources, python, method_args, stages
        )
    return get_stage_tasks(
        model,
        method,
        tool,
        resources,
        python,
        method_args,
        subsets,
        paramsets,
        collect,
    )


def get_pipeline_tasks(
    pipeline: Pipeline,
    method: str,
    tool: Tool,
    resources: dict[str, Any],
    python: Path | str,
    method_args: dict[str, dict[str, Any]],
    stages: list[str] | None,
) -> list[Task]:
    """Get pipeline stage tasks.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline instance.
    method : str
        Name of method to evaluate.
    tool : Tool
        Jobmon tool.
    resources : dict
        Dictionary of compute resources.
    python : Path or str
        Path to Python environment.
    method_args : dict
        Additional keyword arguments passed to stage methods.
    stages : list of str or None
        Name of stages to evaluate. If None, evaluate all pipeline
        stages.

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
                stage,
                method,
                tool,
                resources,
                python,
                method_args.get(stage_name, {}),
                upstream_tasks=upstream_tasks,
            )
            tasks.extend(task_dict[stage_name])

    return tasks


def get_upstream_tasks(
    stage: Stage,
    method: str,
    stage_dict: dict[str, Stage],
    task_dict: dict[str, list[Task]],
    stages: list[str] | None,
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
    stages : list of str or None
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
            # upstream stage is not being evaluated
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
    stage: Stage,
    method: str,
    tool: Tool,
    resources: dict[str, Any],
    python: Path | str,
    method_args: dict[str, Any],
    subsets: dict[str, Any | list[Any]] | None = None,
    paramsets: dict[str, Any | list[Any]] | None = None,
    collect: bool = False,
    upstream_tasks: list[Task] | None = None,
) -> list[Task]:
    """Get stage tasks.

    Parameters
    ----------
    stage : Stage
        Stage instance.
    method : str
        Name of method to evaluate.
    tool : Tool
        Jobmon tool.
    resources : dict
        Dictionary of compute resources.
    python : Path or str
        Path to Python environment.
    method_args : dict
        Additional keyword arguments passed to stage method.
    subsets : dict, optional
        Submodel data subsets to evaluate. If None, evaluate all data
        subsets. Default is None.
    paramsets : dict, optional
        Submodel parameter sets to evaluate. If None, evaluate all
        parameter sets. Default is None.
    collect : bool, optional
        Collect submodel results if `subsets` and `params` are not both
        None. Default is False.
    upstream_tasks : list of Task or None, optional
        List of upstream stage tasks. Default is None.

    Returns
    -------
    list of Task
        List of stage tasks.

    """
    if upstream_tasks is None:
        upstream_tasks = []

    entrypoint = get_entrypoint(python)
    config_path = get_config_path(stage)
    submodel_args = get_submodel_args(stage, method, subsets, paramsets)

    task_template = get_task_template(
        stage.name,
        method,
        tool,
        resources,
        list(method_args.keys()),
        list(submodel_args.keys()),
    )

    if submodel_args:
        tasks = task_template.create_tasks(
            name=f"{stage.name}_{method}",
            upstream_tasks=upstream_tasks,
            max_attempts=1,
            entrypoint=entrypoint,
            config=config_path,
            **method_args,
            **submodel_args,
        )
    else:
        tasks = [
            task_template.create_task(
                name=f"{stage.name}_{method}",
                upstream_tasks=upstream_tasks,
                max_attempts=1,
                entrypoint=entrypoint,
                config=config_path,
                **method_args,
            )
        ]

    if stage.has_submodels and method in stage.collect_after:
        if collect or (subsets is None and paramsets is None):
            # get task for collect method
            tasks.extend(
                get_stage_tasks(
                    stage,
                    "collect",
                    tool,
                    resources,
                    python,
                    method_args,
                    upstream_tasks=tasks,
                )
            )

    return tasks


def get_entrypoint(python: Path | str) -> str:
    """Get path to python entrypoint.

    All stages methods are called via `onemod.main.evaluate()`.

    Parameters
    ----------
    python : Path or str
        Path to python environment.

    Returns
    -------
    str
        Path to python entrypoint.

    """
    return str(Path(python).parent / "onemod")


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


def get_submodel_args(
    stage: Stage,
    method: str,
    subsets: dict[str, Any | list[Any]] | None,
    paramsets: dict[str, Any | list[Any]] | None,
) -> dict[str, Any]:
    """Get dictionary of subset and/or paramset values.

    If stage has submodels and `method` is not 'collect', additional
    args for 'subsets' and/or 'paramsets' are included in the command
    template.

    Parameters
    ----------
    stage : Stage
        Stage instance.
    method : str
        Method being evaluated.
    subsets : dict or None.
        Submodel data subsets to evaluate. If None, evaluate all data
        subsets.
    paramsets : dict or None
        Submodel parameter sets to evaluate. If None, evaluate all
        parameter sets.

    Returns
    -------
    dict
        Dictionary of subset and/or paramset values.

    """
    node_args = {}
    if stage.has_submodels and method != "collect":
        # Get data subsets
        if (filtered_subsets := stage.subsets) is not None:
            if subsets is not None:
                filtered_subsets = stage.get_subset(filtered_subsets, subsets)
            node_args["subsets"] = [
                str(subset)
                for subset in filtered_subsets.to_dict(orient="records")
            ]

        # Get parameter sets
        if (filtered_paramsets := stage.paramsets) is not None:
            if paramsets is not None:
                filtered_paramsets = stage.get_subset(
                    filtered_paramsets, paramsets
                )
            node_args["paramsets"] = [
                str(paramset)
                for paramset in filtered_paramsets.to_dict(orient="records")
            ]

    return node_args


def get_task_template(
    stage_name: str,
    method: str,
    tool: Tool,
    resources: dict[str, Any],
    method_args: list[str],
    submodel_args: list[str],
) -> TaskTemplate:
    """Get stage task template.

    Parameters
    ----------
    stage_name : str
        Stage name.
    method : str
        Name of method being evaluated.
    tool : Tool
        Jobmon tool.
    resources : dict
        Dictionary of compute resources.
    method_args : list of str
        Additional keyword arguments passed to stage method.
    submodel_args : list of str
        List including 'subsets' and/or 'paramsets'.

    Returns
    -------
    TaskTemplate
        Stage task template.

    """
    task_template = tool.get_task_template(
        template_name=f"{stage_name}_{method}",
        command_template=get_command_template(
            stage_name, method, method_args, submodel_args
        ),
        op_args=["entrypoint"],
        task_args=["config"],
        node_args=method_args + submodel_args,
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
    method: str,
    method_args: list[str],
    submodel_args: list[str],
) -> str:
    """Get stage command template.

    All stages methods are called via `onemod.main.evaluate()`. If stage
    has submodels and `method` is not 'collect', additional args for
    'subsets' and/or 'paramsets' are included in the command template.

    Parameters
    ----------
    stage_name : str
        Stage name.
    method : str
        Name of method being evaluated.
    method_args : list of str
        List of additional keyword arguments passed to stage method.
    submodel_args : list of str
        List including 'subsets' and/or 'paramsets'.

    Returns
    -------
    str
        Stage command template.

    """
    command_template = (
        "{entrypoint} --config {config}"
        f" --method {method} --stages {stage_name}"
    )

    for arg in method_args + submodel_args:
        # add 'subsets', 'paramsets', and/or other kwargs
        command_template += f" --{arg} '{{{arg}}}'"

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
