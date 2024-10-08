from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from jobmon.client.api import Tool
        from jobmon.client.task_template import TaskTemplate
    except ImportError:
        pass


def _create_task_template(
    tool: "Tool",
    task_template_name: str,
    node_args: list[str],
    task_args: list[str] | None = None,
    op_args: list[str] | None = None,
    resources_yaml: str | Path | None = None,
) -> "TaskTemplate":
    """Create a Jobmon task template from provided arguments.

    This method creates a Jobmon task template using the provided arguments and returns it.
    Every single task template has a default op_arg that is the entrypoint of the task,
    and keyword args passed to the individual tasks must match what is set on the task template

    Ex. invalid command template:
        {entrypoint} --directory {directory_name}

    Parameters
    ----------
    tool : Tool
        The Jobmon Tool instance to use for creating the task template.
    task_template_name : str
        The name of the task template.
    node_args : list[str]
        The list of node arguments to use for the task template.
    task_args : list[str], optional
        The list of task arguments to use for the task template, by default None.
    op_args : list[str], optional
        The list of operation arguments to use for the task template, by default None.
    resources_yaml : str or Path, optional
        The path to the resources file to use for the task template, by default "".

    Returns
    -------
    TaskTemplate
        The Jobmon task template created using the provided arguments.
    """

    # TODO: This whole method could probably be simplified dramatically by simply
    #  passing in an action and using the stored kwargs to build the command template.
    #  Consider everything besides entrypoint a node_arg for simplicity

    command_template = ["{entrypoint}"]
    if not op_args:
        op_args = []

    if node_args:
        for node_arg in node_args:
            # Rule: the CLI argument must match the name of the arg
            command_template.append(f"--{node_arg}")
            command_template.append(f"{{{node_arg}}}")

    if task_args:
        for task_arg in task_args:
            command_template.append(f"--{task_arg}")
            command_template.append(f"{{{task_arg}}}")

    if op_args:
        for op_arg in op_args:
            command_template.append(f"--{op_arg}")
            command_template.append(f"{{{op_arg}}}")

    # Add in default op_arg
    if "entrypoint" not in op_args:
        op_args.append("entrypoint")

    joined_template = " ".join(command_template)

    template_args = {
        "template_name": task_template_name,
        "command_template": joined_template,
        "node_args": node_args,
        "task_args": task_args,
        "op_args": op_args,
        "default_cluster_name": tool.default_cluster_name,
    }
    if tool.default_cluster_name == "dummy":
        template = tool.get_task_template(
            default_compute_resources={"queue": "null.q"}, **template_args
        )
    else:
        template = tool.get_task_template(
            default_resource_scales={"memory": 0.5, "runtime": 0.5},
            yaml_file=resources_yaml,
            **template_args,
        )

    return template


def create_initialization_template(
    tool: "Tool", task_template_name: str, resources_yaml: str = ""
) -> "TaskTemplate":
    template = _create_task_template(
        tool=tool,
        task_template_name=task_template_name,
        node_args=["stages"],
        task_args=["directory"],
        resources_yaml=resources_yaml,
    )
    return template


def create_modeling_template(
    tool: "Tool", task_template_name: str, resources_yaml: str = ""
) -> "TaskTemplate":
    """Stage modeling template.

    Parameters
    ----------
    tool : Tool
        The Jobmon Tool instance to use for creating the task template.
    task_template_name : str
        The name of the task template.
    resources_yaml : str, optional
        The path to the resources file to use for the task template, by default "".

    Returns
    -------
    TaskTemplate
        The task template for stage modeling.

    """

    # Tasks can be parallelized by an internal concept called submodels
    # swimr_model also uses submodels, but hasn't been implemented yet
    node_args = []
    if task_template_name in [
        "rover_covsel_model",
        "spxmod_model",
        "weave_model",
    ]:
        node_args.append("submodel_id")

    template = _create_task_template(
        tool=tool,
        task_template_name=task_template_name,
        node_args=node_args,
        task_args=["directory"],
        resources_yaml=resources_yaml,
    )

    return template


def create_collection_template(
    tool: "Tool", task_template_name: str, resources_yaml: str = ""
) -> "TaskTemplate":
    """Stage collection template.

    Parameters
    ----------
    task_template_name : str
        The name of the task template.

    Returns
    -------
    TaskTemplate
        The task template for stage collection.

    """

    template = _create_task_template(
        tool=tool,
        task_template_name=task_template_name,
        node_args=["stage_name"],
        task_args=["directory"],
        resources_yaml=resources_yaml,
    )
    return template


def create_deletion_template(
    tool: "Tool", task_template_name: str, resources_yaml: str = ""
) -> "TaskTemplate":
    """Stage deletion template.

    Parameters
    ----------
    tool: Tool
        The Jobmon Tool instance to use for creating the task template.
    task_template_name : str
        The name of the task template.
    resources_yaml : str, optional

    Returns
    -------
    TaskTemplate
        The task template for stage deletion.

    """

    template = _create_task_template(
        tool=tool,
        task_template_name=task_template_name,
        node_args=["result"],
        resources_yaml=resources_yaml,
    )
    return template
