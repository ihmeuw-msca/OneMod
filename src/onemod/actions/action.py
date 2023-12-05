import shutil
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from jobmon.client.task import Task

from onemod.utils import TaskTemplateFactory, TaskRegistry


def callback(func_name):
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

    upstream_actions = order_map[func_name]

    upstream_tasks = []
    for action in upstream_actions:
        tasks = TaskRegistry.get(action.name)
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
    def name(self) -> str:
        return self.func.__name__

    @property
    def task(self) -> "Task":
        """Create a Jobmon task from this action."""
        if self._task is not None:
            return self._task

        # Unpack kwargs into a string for naming purposes
        task_template = TaskTemplateFactory.get_task_template(self.name)
        kwargs_str = "_".join([f"{key}{value}" for key, value in self.kwargs.items()])
        upstream_tasks = self.callback(self.name)
        self._task = task_template.create_task(
            name=f"{self.name}_{kwargs_str}",
            upstream_tasks=upstream_tasks,
            # Requirement: entrypoints map exactly to function names
            entrypoint=shutil.which(self.name),
            **self.kwargs
        )
        # Store the task for later lookup
        TaskRegistry.put(self.name, self._task)
        return self._task

    def evaluate(self):
        """Evaluate the action."""
        return self.func(*self.args, **self.kwargs)
