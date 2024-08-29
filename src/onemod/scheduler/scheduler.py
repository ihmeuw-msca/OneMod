import yaml
from loguru import logger
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Generator

from jobmon.client.workflow import Workflow

from onemod.actions.action import Action
from onemod.actions.data.delete_results import delete_results
from onemod.actions.data.collect_results import (
    collect_results_rover_covsel,
    collect_results_spxmod,
    collect_results_weave
)
from onemod.actions.data.initialize_results import initialize_results
from onemod.actions.models.weave_model import weave_model
from onemod.actions.models.spxmod_model import spxmod_model
from onemod.actions.models.rover_covsel_model import rover_covsel_model
from onemod.application.api import get_application_class
from onemod.scheduler.scheduling_utils import (
    ParentTool,
    SchedulerType,
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
        directory: Path,
        config: OneModConfig,
        stages: list[str],
        default_cluster_name: str = "slurm",
        resources_yaml: str = "",
    ):
        self.directory = directory
        self.config = config
        self.stages = stages
        self.default_cluster_name = default_cluster_name
        self._upstream_task_registry: dict[str, list["Task"]] = {}
        self.resources_yaml = resources_yaml
        self._upstream_task_registry: dict[str, list["Task"]] = {}

    def parent_action_generator(self) -> Generator[Action, None, None]:
        # The schedule always starts with an initialization action
        yield Action(
            initialize_results, stages=self.stages, directory=self.directory
        )
        for stage in self.stages:
            if self.config[stage] is None:
                raise ValueError(f"Error: no settings for stage '{stage}'")
            application_class = get_application_class(stage)
            application = application_class(
                directory=self.directory,
                max_attempts=self.config[stage].max_attempts,
            )
            generator = application.action_generator()
            yield from generator

    def run(self, scheduler_type: SchedulerType) -> None:
        logger.level("DEBUG")
        if scheduler_type == SchedulerType.run_local:
            logger.info("Using local Scheduler")
            for action in self.parent_action_generator():
                action.evaluate()
        else:
            logger.info("Using Jobmon and Slurm")
            ParentTool.initialize_tool(
                default_cluster_name=self.default_cluster_name,
                resources_yaml=self.resources_yaml,
            )
            tool = ParentTool.get_tool()
            workflow = tool.create_workflow()
            tasks = [
                self.create_task_generators(action)
                for action in self.parent_action_generator()
            ]
            workflow.add_tasks(tasks)
            status = workflow.run(configure_logging=True)

            if status != "D":
                # TODO: Summarize errors in workflow
                raise ValueError(
                    f"workflow {workflow.name} failed: {status},"
                    f"Lookup this workflow id {workflow.workflow_id} in your local Jobmon GUI"
                )

    def create_task(self, action: Action) -> "Task":
        """Create a Jobmon task from a given action."""

        # Unpack kwargs into a string for naming purposes
        task_template = TaskTemplateFactory.get_task_template(
            action_name=action.name, resources_yaml=self.resources_yaml
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

    def create_task_generators(self, action: Action) -> "Task":
        """Create a Jobmon task from a given action."""

        logger.debug(f"Creating Task for action: {action.name} over {action.kwargs}")

        # Quick type coercion: the Fire library has strange handling of lists.
        # Force to a string without any spaces
        # This is a catch-all solution but technically only affects initialize_results
        # for arg, value in action.kwargs.items():
        #     if isinstance(value, list):
        #         action.kwargs[arg] = f"[{','.join(value)}]"

        loaded_resources = self.load_resources_from_file(self.resources_path)
        task: Task  # helps with type hinting
        match action.name:
            case "initialize_results":
                task = initialize_results.create_task(
                    compute_resources=loaded_resources,
                    directory=self.directory,
                    stages=self.stages
                )
            case "collect_results_rover_covsel":
                task = collect_results_rover_covsel.create_task(
                    compute_resources=loaded_resources,
                    directory=self.directory)
            case "collect_results_spxmod":
                task = collect_results_spxmod.create_task(
                    compute_resources=loaded_resources,
                    directory=self.directory)
            case "collect_results_weave":
                task = collect_results_weave.create_task(
                    compute_resources=loaded_resources,
                    directory=self.directory
                )
            case "delete_results":
                task = delete_results.create_task(
                    compute_resources=loaded_resources,
                    result=self.directory.joinpath("/results"))
            case "rover_covsel_model":
                task = rover_covsel_model.create_task(
                    compute_resources=loaded_resources,
                    directory=self.directory,
                    submodel_id=action.kwargs["submodel_id"]
                )
            case "weave_model":
                task = weave_model.create_task(
                    compute_resources=loaded_resources,
                    directory=self.directory,
                    submodel_id=action.kwargs["submodel_id"]
                )
            case "spxmod_model":
                task = spxmod_model.create_task(
                    compute_resources=loaded_resources,
                    directory=self.directory,
                    submodel_id=action.kwargs["submodel_id"])
            case _:
                raise ValueError(f"Invalid action name: {action.name}")

        # Store the task for later lookup
        TaskRegistry.put(action.name, task)
        # And connect the upstream tasks
        upstream_tasks = upstream_task_callback(action)
        task.add_upstreams(upstream_tasks)
        logger.debug(f"  Upstreams are {upstream_tasks}")
        return task


    @staticmethod
    def load_resources_from_file(path: str) -> dict:
        with open(path, "r") as stream:
            try:
                loaded_resources = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception(
                    f"Unable to read resources from {path}. "
                    f"Exception: {exc}"
                )
        return loaded_resources
