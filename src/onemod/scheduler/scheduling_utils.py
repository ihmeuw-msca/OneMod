from collections import defaultdict
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING

from onemod.actions.action import Action

if TYPE_CHECKING:
    from jobmon.client.task_template import TaskTemplate
    from jobmon.client.task import Task


class TaskTemplateFactory:
    """
    A helper class to create task templates for each stage and cache the result.
    """

    @classmethod
    def get_task_template(cls, task_template_name: str) -> "TaskTemplate":
        # TODO: Implement
        pass


class TaskRegistry:
    """
    Register tasks on this registry to lookup for upstreams by the action callback.
    """

    # Store on class for global accessibility
    registry: dict[str, set["Task"]] = defaultdict(set)

    @classmethod
    def get(cls, function_name: str):
        return list(cls.registry[function_name])

    @classmethod
    def put(cls, function_name: str, task: "Task"):
        cls.registry[function_name].add(task)


def task_template_cache(func: Callable) -> Callable:
    """Decorator to cache existing task templates by name."""
    cache: dict[str, TaskTemplate] = {}

    @wraps(func)
    def inner_func(*args: Any, **kwargs: Any) -> "TaskTemplate":
        task_template_name = kwargs.get("task_template_name")
        if not task_template_name:
            raise ValueError("task_template_name must be provided")
        if task_template_name in cache:
            return cache[task_template_name]

        template = func(*args, **kwargs)
        cache[task_template_name] = template
        return template

    return inner_func


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
