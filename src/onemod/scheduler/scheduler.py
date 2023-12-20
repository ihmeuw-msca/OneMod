import shutil
from typing import Generator, TYPE_CHECKING

from onemod.actions.action import Action
from onemod.actions.data.initialize_results import initialize_results
from onemod.application.api import get_application_class
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
        experiment_dir: str,
        config: OneModConfig,
        stages: list[str],
        resources_path: str = "",
        default_cluster_name: str = "slurm",
        configure_resources: bool = True,
    ):
        self.experiment_dir = experiment_dir
        self.config = config
        self.stages = stages
        self.resources_path = resources_path
        self.default_cluster_name = default_cluster_name
        self.configure_resources = configure_resources
        self._upstream_task_registry: dict[str, list["Task"]] = {}

    def parent_action_generator(self) -> Generator[Action, None, None]:
        # The schedule always starts with an initialization action
        yield Action(initialize_results, stages=self.stages, experiment_dir=self.experiment_dir)
        for stage in self.stages:
            application_class = get_application_class(stage)
            application = application_class(experiment_dir=self.experiment_dir)
            generator = application.action_generator()
            yield from generator

    def run(self, run_local: bool):
        # TODO: Add args for running with jobmon, i.e. resources file
        if run_local:
            for action in self.parent_action_generator():
                action.evaluate()
        else:
            ParentTool.initialize_tool(
                resources_yaml=self.resources_path,
                default_cluster_name=self.default_cluster_name
            )
            tool = ParentTool.get_tool()
            workflow = tool.create_workflow()
            tasks = [self.create_task(action) for action in self.parent_action_generator()]
            workflow.add_tasks(tasks)
            workflow.run(configure_logging=True)

    def create_task(self, action: Action) -> "Task":
        """Create a Jobmon task from a given action."""

        # Unpack kwargs into a string for naming purposes
        task_template = TaskTemplateFactory.get_task_template(
            action_name=action.name,
            resources_path=self.resources_path
        )
        upstream_tasks = upstream_task_callback(action)
        task = task_template.create_task(
            name=action.name,
            upstream_tasks=upstream_tasks,
            entrypoint=shutil.which(action.entrypoint),
            **action.kwargs
        )
        # Store the task for later lookup
        TaskRegistry.put(action.name, task)
        return task
