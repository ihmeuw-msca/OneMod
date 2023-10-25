"""Create onemod stage tasks."""
from __future__ import annotations

from loguru import logger
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Union

from onemod.orchestration.templates import (
    create_collection_template,
    create_modeling_template,
    create_deletion_template,
)
from onemod.schema.config import ParentConfiguration as GlobalConfig
from onemod.utils import (
    get_rover_covsel_submodels,
    get_swimr_submodels,
    get_weave_submodels,
)

if TYPE_CHECKING:
    from jobmon.client.api import Tool

    from jobmon.client.task import Task


class StageTemplate:
    """Onemod stage template.

    Parameters
    ----------
    stage_name : str
        The name of the stage.
    experiment_dir : Path
        The experiment directory. It must contain config/settings.yml.
    save_intermediate : bool
        Whether to save intermediate stage results.
    cluster_name : str
        Name of the cluster to run the pipeline on.
    resources_file : Union[Path, str]
        Path to the file containing task template resources.
    tool : Tool
        The jobmon Tool instance to use for creating tasks.
    """

    def __init__(
        self,
        stage_name: str,
        config: GlobalConfig,
        experiment_dir: Union[Path, str],
        save_intermediate: bool,
        cluster_name: str,
        resources_file: Union[Path, str],
        tool: "Tool",
    ) -> None:
        """Create onemod stage template."""
        self.stage_name = stage_name
        self.experiment_dir = Path(experiment_dir)
        self.stage_dir = self.experiment_dir / "results" / stage_name
        self.save_intermediate = save_intermediate
        self.cluster_name = cluster_name
        self.tool = tool
        self.config = config
        self.resources_file = Path(resources_file)

        # Get stage submodels
        self.submodel_ids = None
        if stage_name == "rover_covsel":
            self.submodel_ids = get_rover_covsel_submodels(experiment_dir)
        elif stage_name == "swimr":
            self.submodel_ids = get_swimr_submodels(experiment_dir)
        elif stage_name == "weave":
            self.submodel_ids = get_weave_submodels(experiment_dir)

    def create_tasks(self, upstream_tasks: list["Task"]) -> list["Task"]:
        """Create stage tasks.

        Parameters
        ----------
        upstream_tasks : list of Task
            List of upstream tasks for the current stage.

        Returns
        -------
        list of Task
            List of tasks representing the current stage.

        """

        # Create stage modeling tasks
        # Ensemble and regmod_smooth aren't parallelized, the rest are. No submodel concepts needed for ensemble and smoothing.
        parallel = self.submodel_ids is not None
        modeling_tasks = self.create_modeling_tasks(
            max_attempts=self.config.max_attempts,
            upstream_tasks=upstream_tasks,
            parallel=parallel
        )
        # Ensemble is the terminal stage so no collection task is needed.
        if self.stage_name in ["ensemble"]:
            return [modeling_tasks]

        # Create stage collection task
        collection_task = self.create_collection_task(upstream_tasks=modeling_tasks)

        tasks = [*modeling_tasks, collection_task]

        # Swimr can be highly parallel and generate lots of IO, so optionally add deletion tasks here
        if not self.save_intermediate and self.stage_name == "swimr":
            tasks.extend(self.create_deletion_tasks(upstream_tasks=[collection_task]))
        return tasks

    def create_modeling_tasks(
        self, max_attempts: int, upstream_tasks: list["Task"], parallel: bool
    ) -> list["Task"]:
        """Create stage modeling tasks.

        Parameters
        ----------
        max_attempts : int
            The maximum number of attempts for each modeling task.
        upstream_tasks : list of Task
            List of upstream tasks for the modeling tasks.
        parallel: bool
            Whether this task template has multiple tasks. If so, can parallelize by adding submodel_id node arg.

        Returns
        -------
        list of Task
            List of tasks representing the modeling stage.

        """
        model_template = create_modeling_template(
            tool=self.tool,
            task_template_name=f"{self.stage_name}_modeling_template",
            resources_path=self.resources_file,
            parallel=parallel,
        )

        model_task_args = {
            "entrypoint": shutil.which(f"{self.stage_name}_model"),
            "experiment_dir": str(self.experiment_dir),
        }

        # TODO: Something that should be fixed in Jobmon, the create_tasks method returns an empty list if called with no node_args.
        # Use create_task as a workaround in non parallel cases.
        if parallel:
            model_task_args["submodel_id"] = self.submodel_ids
            create_tasks_callable = model_template.create_tasks
        else:
            # Wrap in a lambda function since create_task returns a Task, not a list. For type consistency we want a single valued list
            create_tasks_callable = lambda **kwargs: [model_template.create_task(**kwargs)]

        tasks = create_tasks_callable(
            name=f"{self.stage_name}_modeling_tasks",
            max_attempts=max_attempts,
            upstream_tasks=upstream_tasks,
            **model_task_args
        )
        return tasks

    def create_collection_task(self, upstream_tasks: list["Task"]) -> "Task":
        """Create stage collection task.

        Parameters
        ----------
        upstream_tasks : list of Task
            List of upstream tasks for the collection task.

        Returns
        -------
        Task
            The collection task.

        """
        collection_template = create_collection_template(
            tool=self.tool,
            task_template_name=f"collection_template",
            resources_path=self.resources_file,
        )
        return collection_template.create_task(
            name=f"{self.stage_name}_collection_task",
            max_attempts=2,
            upstream_tasks=upstream_tasks,
            entrypoint=shutil.which("collect_results"),
            stage_name=self.stage_name,
            experiment_dir=self.experiment_dir,
        )

    def create_deletion_tasks(self, upstream_tasks: list["Task"]) -> list["Task"]:
        """Create stage deletion tasks.

        Parameters
        ----------
        upstream_tasks : list of Task
            List of upstream tasks for the deletion tasks.

        Returns
        -------
        list of Task
            List of tasks representing the deletion stage.

        """
        tasks = []

        # Delete submodels
        if not self.submodel_ids:
            logger.warning(f"This stage {self.stage_name} has no submodels to delete, skipping deletion task creation.")
            return []

        submodel_template = create_deletion_template(
            tool=self.tool,
            task_template_name=f"submodel_deletion_template",
            resources_path=self.resources_file,
        )
        tasks.extend(
            submodel_template.create_tasks(
                name=f"{self.stage_name}_submodel_deletion_task",
                max_attempts=1,
                upstream_tasks=upstream_tasks,
                entrypoint=shutil.which("delete_results"),
                result=[
                    self.stage_dir / "submodels" / submodel_id
                    for submodel_id in self.submodel_ids
                ],
            )
        )

        return tasks
