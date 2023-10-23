from pathlib import Path
from typing import TYPE_CHECKING

from onemod.utils import task_template_cache


if TYPE_CHECKING:
    from jobmon.client.task_template import TaskTemplate
    from jobmon.client.api import Tool


@task_template_cache
def create_task_template(
    tool: Tool,
    task_template_name: str,
    node_args: list[str],
    task_args: list[str] | None = None,
    op_args: list[str] | None = None,
    resources_path: str | Path | None = None,
) -> TaskTemplate:
    """Create a Jobmon task template from provided arguments.

    This method creates a Jobmon task template using the provided arguments and returns it.
    Every single task template has a default op_arg that is the entrypoint of the task,
    and keyword args passed to the individual tasks must match what is set on the task template.

    Ex. invalid command template:
        {entrypoint} --experiment_dir {directory_name}

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
    resources_path : str or Path, optional
        The path to the resources file to use for the task template, by default "".

    Returns
    -------
    TaskTemplate
        The Jobmon task template created using the provided arguments.
    """

    command_template = ["{entrypoint}"]

    for node_arg in node_args:
        # Rule: the CLI argument must match the name of the arg
        command_template.append(f"{node_arg}")
        command_template.append(f"--{{{node_arg}}}")

    if task_args is not None:
        for task_arg in task_args:
            command_template.append(f"{task_arg}")
            command_template.append(f"--{{{task_arg}}}")

    if op_args is not None:
        for op_arg in op_args:
            command_template.append(f"{op_arg}")
            command_template.append(f"--{{{op_arg}}}")

    joined_template = " ".join(command_template)

    template = tool.get_task_template(
        template_name=task_template_name,
        command_template=joined_template,
        node_args=node_args,
        task_args=task_args,
        op_args=op_args,
        default_cluster_name=tool.default_cluster_name,
        yaml_file=resources_path,
    )

    return template



# def create_modeling_template(tool: Tool, task_template_name: str, resources_path: str | Path) -> TaskTemplate:
#     """Stage modeling template.

#     Parameters
#     ----------
#     task_template_name : str
#         The name of the task template.

#     Returns
#     -------
#     TaskTemplate
#         The task template for stage modeling.

#     """
#     template = _create_task_template(
#         tool=tool,
#         task_template_name=task_template_name,
#         node_args=["submodel_id"],
#         task_args=["experiment_dir"],
#         op_args=["entrypoint"],
#         resources_path=resources_path,
#     )


#     tool.get_task_template(
#         template_name=task_template_name,
#         command_template="{entrypoint}"
#         " --experiment_dir {experiment_dir}"
#         " --submodel_id {submodel_id}",
#         node_args=["submodel_id"],
#         task_args=["experiment_dir"],
#         op_args=["entrypoint"],
#         default_cluster_name=tool.default_cluster_name,
#     )

#     template.set_default_compute_resources_from_yaml(
#         default_cluster_name=tool.default_cluster_name,
#         yaml_file=resources_path,
#     )
#     return template

# @task_template_cache(task_template_name="collection_template")
# def create_collection_template(self, task_template_name: str) -> TaskTemplate:
#     """Stage collection template.

#     Parameters
#     ----------
#     task_template_name : str
#         The name of the task template.

#     Returns
#     -------
#     TaskTemplate
#         The task template for stage collection.

#     """
#     template = self.tool.get_task_template(
#         template_name=task_template_name,
#         command_template="{entrypoint} {stage_name}"
#         " --experiment_dir {experiment_dir}",
#         node_args=["stage_name"],
#         task_args=["experiment_dir"],
#         op_args=["entrypoint"],
#         default_cluster_name=self.cluster_name,
#     )
#     if task_template_name in self.resources:
#         template.set_default_compute_resources_from_dict(
#             cluster_name=self.cluster_name,
#             compute_resources=self.resources[task_template_name][self.cluster_name],
#         )
#     return template

# @task_template_cache(task_template_name="deletion_template")
# def create_deletion_template(self, task_template_name: str) -> TaskTemplate:
#     """Stage deletion template.

#     Parameters
#     ----------
#     task_template_name : str
#         The name of the task template.

#     Returns
#     -------
#     TaskTemplate
#         The task template for stage deletion.

#     """
#     template = self.tool.get_task_template(
#         template_name=task_template_name,
#         command_template="{entrypoint} stage"
#         " --experiment_dir {experiment_dir}"
#         " --stage_name {stage_name}",
#         node_args=["stage_name"],
#         task_args=["experiment_dir"],
#         op_args=["entrypoint"],
#         default_cluster_name=self.cluster_name,
#     )
#     if task_template_name in self.resources:
#         template.set_default_compute_resources_from_dict(
#             cluster_name=self.cluster_name,
#             compute_resources=self.resources[task_template_name][self.cluster_name],
#         )
#     return template




# def create_submodel_deletion_template(
#     self, task_template_name: str
# ) -> TaskTemplate:
#     """Stage submodel deletion template.

#     Parameters
#     ----------
#     task_template_name : str
#         The name of the task template.

#     Returns
#     -------
#     TaskTemplate
#         The task template for stage submodel deletion.

#     """
#     template = self.tool.get_task_template(
#         template_name=task_template_name,
#         command_template="{entrypoint} result --result {result}",
#         node_args=["result"],
#         op_args=["entrypoint"],
#         default_cluster_name=self.cluster_name,
#     )
#     if task_template_name in self.resources:
#         template.set_default_compute_resources_from_dict(
#             cluster_name=self.cluster_name,
#             compute_resources=self.resources[task_template_name][self.cluster_name],
#         )
#     return template
