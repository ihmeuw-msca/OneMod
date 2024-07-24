from collections import defaultdict
from enum import StrEnum
from typing import TYPE_CHECKING

try:
    from jobmon.client.api import Tool
    from jobmon.client.task import Task
    from jobmon.client.task_template import TaskTemplate
except ImportError:
    pass


from onemod.actions.action import Action
from onemod.scheduler.templates import (
    create_collection_template,
    create_deletion_template,
    create_initialization_template,
    create_modeling_template,
)


SchedulerType = StrEnum("SchedulerType", ['jobmon', 'run_local'])

class ParentTool:
    """Singleton implementation of a single tool used across OneMod for scheduling."""

    # Quoting will prevent errors if Jobmon is not installed
    tool: "Tool" = None

    @classmethod
    def initialize_tool(
        cls,
        resources_yaml: str,
        default_cluster_name: str,
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
    """
    Given an action, we should know (based on the action name) what the relevant upstream tasks
    are.
    The algorithm: as tasks are created, we will add them to a global registry keyed by action
    name. Actions should know exactly what their upstream tasks are.
    Assumes tasks are added in the right dependency order, but that's a requirement
    for local execution anyway.
    """

    order_map = {
        "initialize_results": [],
        "rover_covsel_model": ["initialize_results"],
        "spxmod_model": ["collect_results", "initialize_results"],
        "weave_model": [
            "collect_results",
            "collect_results",
            "initialize_results",
        ],
        "ensemble_model": 3 * ["collect_results"] + ["initialize_results"],
        # Logic for collect results: set all modeling tasks as dependencies.
        # Due to traversal order of the generator, the rover collection task must be created
        # prior to weave modeling tasks being instantiated, therefore this is
        # theoretically safe to do.
        # Vice versa: when spxmod_model's task is created, there can be at most one
        # previously created collect task (for rover)
        "collect_results": [
            "rover_covsel_model",
            "spxmod_model",
            "weave_model",
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
