from collections import defaultdict
from typing import TYPE_CHECKING

from jobmon.client.api import Tool

from onemod.actions.action import Action
from onemod.scheduler.templates import (
    create_collection_template,
    create_deletion_template,
    create_initialization_template,
    create_modeling_template,
)

if TYPE_CHECKING:
    from jobmon.client.task import Task
    from jobmon.client.task_template import TaskTemplate


class ParentTool:
    """Singleton implementation of a single tool used across OneMod for scheduling."""

    tool: Tool | None = None

    @classmethod
    def initialize_tool(
        cls, resources_yaml: str, default_cluster_name: str
    ) -> None:
        if cls.tool is None:
            cls.tool = Tool(name="onemod_tool")
            cls.tool.set_default_cluster_name(default_cluster_name)
            cls.tool.set_default_compute_resources_from_yaml(
                default_cluster_name=default_cluster_name,
                yaml_file=resources_yaml,
            )

    @classmethod
    def get_tool(cls) -> "Tool":
        if cls.tool is None:
            raise ValueError("Tool has not been initialized")
        return cls.tool


class TaskTemplateFactory:
    """
    A helper class to create task templates for each stage.
    """

    @classmethod
    def get_task_template(
        cls,
        action_name: str,
        resources_path: str = "",
        configure_resources: bool = True,
    ) -> "TaskTemplate":
        tool = ParentTool.get_tool()

        if action_name == "initialize_results":
            task_template_callable = create_initialization_template
        elif "collect" in action_name:
            task_template_callable = create_collection_template
        elif action_name == "delete_results":
            task_template_callable = create_deletion_template
        elif "model" in action_name:
            task_template_callable = create_modeling_template
        else:
            raise ValueError(f"Invalid action name: {action_name}")

        task_template = task_template_callable(
            tool=tool,
            task_template_name=action_name,
            resources_path=resources_path,
            configure_resources=configure_resources,
        )

        return task_template


class TaskRegistry:
    """
    Register tasks on this registry to lookup for upstreams by the action callback.
    """

    # Store on class for global accessibility
    # Keys are function (action) names, values are sets of tasks for that action
    registry: defaultdict[str, set["Task"]] = defaultdict(set)

    @classmethod
    def get(cls, function_name: str) -> list["Task"]:
        return list(cls.registry[function_name])

    @classmethod
    def put(cls, function_name: str, task: "Task") -> None:
        cls.registry[function_name].add(task)


def upstream_task_callback(action: Action) -> list["Task"]:
    """Return upstream tasks for a given action.

    Assumes tasks are added in the correct dependency order.

    Notes
    -----
    * Assumes tasks are added in the correct dependency order.
    * Logic for setting all modeling tasks as `collect_results`
      dependencies: Due to traversal order of the generator, the rover
      collection task must be created prior to weave modeling task being
      instantiated, therefore this is theoretically safe to do. Vice
      versa: when spxmod_model's task is created, there can be at most
      one previously created collect task (from rover).

    # FIXME: The logic above was fine when no stages could be run in
      parallel, but now we should change things so that kreg doesn't
      need to wait for weave models to finish. This will also be true if
      we ever wanted to use the delete_results task again.

    """

    order_map = {
        "initialize_results": [],
        "rover_covsel_model": ["initialize_results"],
        "spxmod_model": ["initialize_results", "collect_results"],
        "weave_model": ["initialize_results", "collect_results"],
        "kreg_model": ["initialize_results", "collect_results"],
        "ensemble_model": ["initialize_results", "collect_results"],
        "collect_results": [
            "rover_covsel_model",
            "spxmod_model",
            "weave_model",
            "kreg_model",
        ],
    }
    func_name = action.name
    upstream_action_names = order_map[func_name]

    upstream_tasks = []
    for action_name in upstream_action_names:
        tasks = TaskRegistry.get(action_name)
        if any(tasks):
            upstream_tasks.extend(tasks)
    return upstream_tasks
