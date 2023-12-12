from collections import defaultdict
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from jobmon.client.task_template import TaskTemplate
    from jobmon.client.task import Task


class TaskTemplateFactory:
    """
    A helper class to create task templates for each stage and cache the result.
    """

    @classmethod
    def get_task_template(cls, task_template_name: str) -> TaskTemplate:
        # TODO: Implement
        pass


class TaskRegistry:
    """
    Register tasks on this registry to lookup for upstreams by the action callback.
    """

    registry: dict[str, set["Task"]] = defaultdict(set)  # Store on class for accessibility

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
    def inner_func(*args: Any, **kwargs: Any) -> TaskTemplate:
        task_template_name = kwargs.get("task_template_name")
        if not task_template_name:
            raise ValueError("task_template_name must be provided")
        if task_template_name in cache:
            return cache[task_template_name]

        template = func(*args, **kwargs)
        cache[task_template_name] = template
        return template

    return inner_func


def callback(action: Action):
    """
    Idea: we will have a global data structure which is a dict of tasks by function names.
    We will be able to look up known upstream tasks that have been added to the registry.

    Assumes tasks are added in roughly the right dependency order, but that's a requirement
    for local execution anyways.
    """
    order_map = {
        # TODO: Complete this order map
        # For the first task of the stage, we need to be able to optionally lookup prior
        # stages that may or may not be specified in the config.

        # I.e. weave_model may not depend on regmod_smooth if we have stages=['weave']
        'rover_covsel_model': ['initialize_results'],
        'collect_results': ['rover_covsel_model'],
    }
    func_name = action.name
    upstream_actions = order_map[func_name]

    upstream_tasks = []
    for action in upstream_actions:
        tasks = TaskRegistry.get(action.name)
        upstream_tasks.extend(tasks)
    return upstream_tasks