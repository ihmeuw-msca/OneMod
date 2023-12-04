import shutil
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from jobmon.client.task_template import TaskTemplate
    from jobmon.client.task import Task


class Action:
    """Decorator for actions.

    Allows local execution as well as instantiation of a Jobmon task.
    """

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, *args, **kwargs):
        """Evaluate the function locally with provided arguments."""
        return self.func(*args, **kwargs)

    def task(
        self,
        task_template: "TaskTemplate",
        upstream_tasks: list["Task"],
        **kwargs
    ) -> "Task":
        """Create a Jobmon task from this action."""

        # Unpack kwargs into a string for naming purposes
        kwargs_str = "_".join([f"{key}{value}" for key, value in kwargs.items()])
        return task_template.create_task(
            name=f"{self.func.__name__}_{kwargs_str}",
            upstream_tasks=upstream_tasks,
            # Requirement: entrypoints map exactly to function names
            entrypoint=shutil.which(self.func.__name__),
            **kwargs
        )