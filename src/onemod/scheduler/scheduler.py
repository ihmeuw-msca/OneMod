import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Generator

from onemod.actions.action import Action
from onemod.actions.data.initialize_results import initialize_results
from onemod.application.api import get_application_class
from onemod.scheduler.scheduling_utils import (
    ParentTool,
    TaskRegistry,
    TaskTemplateFactory,
    upstream_task_callback,
    SchedulerType
)
from onemod.schema import OneModConfig

if TYPE_CHECKING:
    try:
        from jobmon.client.task import Task
    except ImportError:
        pass



class Scheduler:
    def __init__(
        self,
        directory: str | Path,
        config: OneModConfig,
        stages: list[str],
        resources_path: str = "",
        default_cluster_name: str = "slurm",
        configure_resources: bool = True,
    ):
        self.directory = directory
        self.config = config
        self.stages = stages
        self.resources_path = resources_path
        self.default_cluster_name = default_cluster_name
        self.configure_resources = configure_resources
        self._upstream_task_registry: dict[str, list["Task"]] = {}

    def parent_action_generator(self) -> Generator[Action, None, None]:
        # The schedule always starts with an initialization action
        yield Action(
            initialize_results, stages=self.stages, directory=self.directory
        )
        for stage in self.stages:
            application_class = get_application_class(stage)
            application = application_class(
                directory=self.directory,
                max_attempts=self.config[stage].max_attempts,
            )
            generator = application.action_generator()
            yield from generator

    def run(self, scheduler_type: SchedulerType) -> None:
        # TODO: Add args for running with jobmon, i.e. resources file
        if scheduler_type == SchedulerType.run_local:
            for action in self.parent_action_generator():
                action.evaluate()
        else:
            ParentTool.initialize_tool(
                resources_yaml=self.resources_path,
                default_cluster_name=self.default_cluster_name,
            )
            tool = ParentTool.get_tool()
            workflow = tool.create_workflow()
            tasks = [
                self.create_task(action)
                for action in self.parent_action_generator()
            ]
            workflow.add_tasks(tasks)
            status = workflow.run(configure_logging=True)

            if status != "D":
                # TODO: Summarize errors in workflow
                raise ValueError(f"workflow {workflow.name} failed: {status}")

    def create_task(self, action: Action) -> "Task":
        """Create a Jobmon task from a given action."""

        # Unpack kwargs into a string for naming purposes
        task_template = TaskTemplateFactory.get_task_template(
            action_name=action.name,
            resources_path=self.resources_path,
            configure_resources=self.configure_resources,
        )
        upstream_tasks = upstream_task_callback(action)

        # Quick type coercion: the Fire library has strange handling of lists.
        # Force to a string without any spaces
        # This is a catch-all solution but technically only affects initialize_results
        for arg, value in action.kwargs.items():
            if isinstance(value, list):
                action.kwargs[arg] = f"[{','.join(value)}]"

        task = task_template.create_task(
            name=action.name,
            upstream_tasks=upstream_tasks,
            entrypoint=shutil.which(action.entrypoint),
            **action.kwargs,
        )
        # Store the task for later lookup
        TaskRegistry.put(action.name, task)
        return task
