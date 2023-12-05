import functools
import shutil
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from jobmon.client.task_template import TaskTemplate
    from jobmon.client.task import Task

from onemod.utils import TaskTemplateFactory


def callback(func_name):
    order_map = {
        'rover_covsel_model': ['initialize_results'],
        'collect_results': ['rover_covsel_model'],
    }

    upstream_actions = order_map[func_name]

    upstream_tasks = []
    for action in upstream_actions:
        tasks = callback_dict[action]
        upstream_tasks.extend(tasks)
    return upstream_tasks


class Action:
    """Wrapper for actions.

    Allows local execution as well as instantiation of a Jobmon task.

    An action has 2 key use cases:
    1. Be run as a callable
    2. Be added to a Jobmon task dag
    """

    def __init__(self, func: Callable, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.callback = callback  # For setting upstreams
        self._task = None  # Compute and store once

    @property
    def task(self) -> "Task":
        """Create a Jobmon task from this action."""
        if self._task is not None:
            return self._task

        # Unpack kwargs into a string for naming purposes
        task_template = TaskTemplateFactory.get_task_template(self.func.__name__)
        kwargs_str = "_".join([f"{key}{value}" for key, value in self.kwargs.items()])
        upstream_tasks = self.callback(self.func.__name__)
        self._task = task_template.create_task(
            name=f"{self.func.__name__}_{kwargs_str}",
            upstream_tasks=upstream_tasks,
            # Requirement: entrypoints map exactly to function names
            entrypoint=shutil.which(self.func.__name__),
            **self.kwargs
        )
        return self._task

    def evaluate(self):
        """Evaluate the action."""
        return self.func(*self.args, **self.kwargs)
