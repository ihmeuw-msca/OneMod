import shutil
from typing import Generator, TYPE_CHECKING

from onemod.actions.action import Action
from onemod.scheduler.scheduling_utils import upstream_task_callback, TaskRegistry, TaskTemplateFactory

if TYPE_CHECKING:
    from jobmon.client.task import Task


class Scheduler:

    def __init__(self, stages: list[str]):
        self.stages = stages
        self._upstream_task_registry: dict[str, list["Task"]] = {}

    def parent_action_generator(self) -> Generator[Action, None, None]:
        for stage in self.stages:
            application = get_application(stage)  # TODO: A simple lookup table should suffice
            generator = application.action_generator()
            yield from generator

    def run(self, run_local: bool):
        if run_local:
            for action in self.parent_action_generator():
                action.evaluate()
        else:
            workflow = self.create_workflow()
            tasks = [self.create_task(action) for action in self.parent_action_generator()]
            workflow.add_tasks(tasks)
            workflow.run(configure_logging=True)

    def get_upstream_tasks(self, action: Action):
        """
        Compute and cache the upstreams for a given action.
        """
        upstream_tasks = self._upstream_task_registry[action.name]
        if upstream_tasks:
            return upstream_tasks

        upstream_tasks = upstream_task_callback(action)
        self._upstream_task_registry[action.name] = upstream_tasks
        return upstream_tasks

    def create_task(self, action: Action) -> "Task":
        """Create a Jobmon task from a given action."""

        # Unpack kwargs into a string for naming purposes
        task_template = TaskTemplateFactory.get_task_template(action.name)
        kwargs_str = "_".join([f"{key}{value}" for key, value in action.kwargs.items()])
        upstream_tasks = self.get_upstream_tasks(action)
        task = task_template.create_task(
            name=f"{action.name}_{kwargs_str}",
            upstream_tasks=upstream_tasks,
            # Requirement: entrypoints map exactly to function names
            entrypoint=shutil.which(action.name),
            **action.kwargs
        )
        # Store the task for later lookup
        TaskRegistry.put(action.name, task)
        return action._task
