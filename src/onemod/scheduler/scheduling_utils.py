from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional, TYPE_CHECKING

from onemod.actions.action import Action
from onemod.scheduler.templates import (
    create_collection_template,
    create_deletion_template,
    create_initialization_template,
    create_modeling_template,
)

if TYPE_CHECKING:
    from jobmon.client.api import Tool
    from jobmon.client.task_template import TaskTemplate
    from jobmon.client.task import Task


class ParentTool:
    """Singleton implementation of a single tool used across OneMod for scheduling."""

    tool: Optional["Tool"] = None

    @classmethod
    def initialize_tool(cls, resources_yaml: str, default_cluster_name: str) -> None:
        if cls.tool is None:
            cls.tool = Tool(name="onemod_tool")
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
    def get_task_template(cls, action_name: str, resources_path: str = "") -> "TaskTemplate":

        tool = ParentTool.get_tool()

        if action_name == "initialize_results":
            task_template = create_initialization_template(tool=tool, resources_path=resources_path)
        elif action_name == "collect_results":
            task_template = create_collection_template(tool=tool, resources_path=resources_path)
        elif action_name == "delete_results":
            task_template = create_deletion_template(tool=tool, resources_path=resources_path)
        elif "model" in action_name:
            task_template = create_modeling_template(tool=tool, resources_path=resources_path)
        else:
            raise ValueError(f"Invalid action name: {action_name}")

        return task_template

class TaskRegistry:
    """
    Register tasks on this registry to lookup for upstreams by the action callback.
    """
    # Store on class for global accessibility
    # Keys are function (action) names, values are sets of tasks for that action
    registry: defaultdict[str, set["Task"]] = defaultdict(set)

    @classmethod
    def get(cls, function_name: str):
        return list(cls.registry[function_name])

    @classmethod
    def put(cls, function_name: str, task: "Task"):
        cls.registry[function_name].add(task)


def upstream_task_callback(action: Action) -> list["Task"]:
    """
    Given an action, we should know (based on the action name) what the relevant upstream tasks
    are.
    The algorithm: as tasks are created, we will add them to a global registry keyed by action
    name. Actions should know exactly what their upstream tasks are.
    Assumes tasks are added in the right dependency order, but that's a requirement
    for local execution anyway.
    """

    order_map = {
        'initialize_results': [],
        'rover_covsel_model': ['initialize_results'],
        'collect_results_rover_covsel': ['rover_covsel_model'],
        'regmod_smooth_model': ['collect_results_rover_covsel', 'initialize_results'],
        'collect_results_regmod_smooth': ['regmod_smooth_model'],
        'weave_model': ['collect_results_regmod_smooth', 'collect_results_rover_covsel', 'initialize_results'],
        'collect_results_weave': ['weave_model'],
    }
    func_name = action.name
    upstream_action_names = order_map[func_name]

    upstream_tasks = []
    for action_name in upstream_action_names:
        tasks = TaskRegistry.get(action_name)
        if any(tasks):
            upstream_tasks.extend(tasks)
    return upstream_tasks