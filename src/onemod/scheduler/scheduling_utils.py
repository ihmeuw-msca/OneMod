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