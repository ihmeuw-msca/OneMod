from typing import Generator, TYPE_CHECKING

from onemod.actions.action import Action
from onemod.scheduler.scheduling_utils import (
    ParentTool,
    TaskRegistry,
    TaskTemplateFactory,
    upstream_task_callback,
)
from onemod.schema.models.api import OneModConfig

if TYPE_CHECKING:
    from jobmon.client.task import Task


class Scheduler:

    def __init__(
        self,
        config: OneModConfig,
        stages: list[str],
        resources_yaml: str = "",
        default_cluster_name: str = "slurm",
        configure_resources: bool = True,
    ):
        self.config = config
        self.stages = stages
        self.resources_yaml = resources_yaml
        self.default_cluster_name = default_cluster_name
        self.configure_resources = configure_resources
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
            tasks = [action.task for action in self.parent_action_generator()]
            workflow.add_tasks(tasks)
            workflow.run(configure_logging=True)

    @property
    def create_task(self, action: Action) -> "Task":
        """Create a Jobmon task from a given action."""

        # Unpack kwargs into a string for naming purposes
        task_template = TaskTemplateFactory.get_task_template(action.name)
        kwargs_str = "_".join([f"{key}{value}" for key, value in action.kwargs.items()])
        upstream_tasks = callback(action)
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
