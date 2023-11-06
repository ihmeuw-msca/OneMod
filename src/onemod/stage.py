"""Create onemod stage tasks."""
from __future__ import annotations

from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Union

from onemod.schema.models.parent_config import ParentConfiguration as GlobalConfig
from onemod.utils import (
    get_rover_covsel_submodels,
    get_swimr_submodels,
    get_weave_submodels,
    load_settings,
    task_template_cache,
)

if TYPE_CHECKING:
    from jobmon.client.api import Tool
    from jobmon.client.task_template import TaskTemplate
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
        experiment_dir: Union[Path, str],
        save_intermediate: bool,
        cluster_name: str,
        resources_file: Union[Path, str],
        tool: Tool,
    ) -> None:
        """Create onemod stage template."""
        self.stage_name = stage_name
        self.experiment_dir = Path(experiment_dir)
        self.stage_dir = self.experiment_dir / "results" / stage_name
        self.save_intermediate = save_intermediate
        self.cluster_name = cluster_name
        self.tool = tool

        # Get task resources
        resources = load_settings(resources_file, raise_on_error=False, as_model=False)
        if "task_template_resources" in resources:
            self.resources = resources["task_template_resources"]
        else:
            self.resources = {}

        # Get stage submodels
        if stage_name == "rover_covsel":
            self.submodel_ids = get_rover_covsel_submodels(experiment_dir)
        elif stage_name == "swimr":
            self.submodel_ids = get_swimr_submodels(experiment_dir)
        elif stage_name == "weave":
            self.submodel_ids = get_weave_submodels(experiment_dir)
        else:
            self.submodel_ids = ["dummy"]

    def create_tasks(self, upstream_tasks: list[Task]) -> list[Task]:
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
        config = load_settings(
            self.experiment_dir / "config" / "settings.yml", as_model=True
        )

        # Create stage initialization tasks
        initialization_tasks = self.create_initialization_task()

        # Create stage modeling tasks
        modeling_tasks = self.create_modeling_tasks(
            max_attempts=config.max_attempts,
            upstream_tasks=upstream_tasks + [initialization_tasks[-1]],
        )
        # Ensemble and regmod_smooth aren't parallelized,
        # so no need to implement collection or deletion tasks.
        if self.stage_name in ["ensemble"]:
            return [*initialization_tasks, *modeling_tasks]

        # Create stage collection task
        collection_task = self.create_collection_task(upstream_tasks=modeling_tasks)

        # Create optional stage deletion tasks
        tasks = [*initialization_tasks, *modeling_tasks, collection_task]
        if self.save_intermediate or self.stage_name == "regmod_smooth":
            return tasks
        tasks.extend(self.create_deletion_tasks(upstream_tasks=[collection_task]))
        return tasks

    def create_initialization_task(self) -> list[Task]:
        """Create stage initialization tasks.

        Returns
        -------
        list of Task
            List of tasks for stage initialization.

        """
        tasks = []

        # Delete submodels
        if (
            self.stage_name in ["rover", "swimr"]
            and (self.stage_dir / "submodels").is_dir()
        ):
            submodel_template = self.create_submodel_deletion_template(
                task_template_name=f"{self.stage_name}_submodel_initialization_template"
            )
            tasks.extend(
                submodel_template.create_tasks(
                    name=f"{self.stage_name}_submodel_initialization_task",
                    max_attempts=1,
                    entrypoint=shutil.which("delete_results"),
                    result=list((self.stage_dir / "submodels").iterdir()),
                )
            )

        # Initialize directories
        initialization_template = self.create_initialization_template()
        tasks.append(
            initialization_template.create_task(
                name=f"{self.stage_name}_initialization_task",
                max_attempts=1,
                upstream_tasks=tasks if len(tasks) > 0 else None,
                entrypoint=shutil.which("initialize_results"),
                stage_name=self.stage_name,
                experiment_dir=self.experiment_dir,
            )
        )
        return tasks

    def create_modeling_tasks(
        self, max_attempts: int, upstream_tasks: list[Task]
    ) -> list[Task]:
        """Create stage modeling tasks.

        Parameters
        ----------
        max_attempts : int
            The maximum number of attempts for each modeling task.
        upstream_tasks : list of Task
            List of upstream tasks for the modeling tasks.

        Returns
        -------
        list of Task
            List of tasks representing the modeling stage.

        """
        model_template = self.create_modeling_template(
            task_template_name=f"{self.stage_name}_modeling_template"
        )
        return model_template.create_tasks(
            name=f"{self.stage_name}_modeling_tasks",
            max_attempts=max_attempts,
            upstream_tasks=upstream_tasks,
            entrypoint=shutil.which(f"{self.stage_name}_model"),
            experiment_dir=self.experiment_dir,
            submodel_id=self.submodel_ids,
        )

    def create_collection_task(self, upstream_tasks: list[Task]) -> Task:
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
        collection_template = self.create_collection_template()
        return collection_template.create_task(
            name=f"{self.stage_name}_collection_task",
            max_attempts=2,
            upstream_tasks=upstream_tasks,
            entrypoint=shutil.which("collect_results"),
            stage_name=self.stage_name,
            experiment_dir=self.experiment_dir,
        )

    def create_deletion_tasks(self, upstream_tasks: list[Task]) -> list[Task]:
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
        if self.stage_name in ["rover", "swimr"]:
            submodel_template = self.create_submodel_deletion_template(
                task_template_name=f"{self.stage_name}_submodel_deletion_template"
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

        # Delete intermediate result directories
        deletion_template = self.create_deletion_template()
        tasks.append(
            deletion_template.create_task(
                name=f"{self.stage_name}_deletion_task",
                max_attempts=1,
                upstream_tasks=tasks if len(tasks) > 0 else upstream_tasks,
                entrypoint=shutil.which("delete_results"),
                stage_name=self.stage_name,
                experiment_dir=self.experiment_dir,
            )
        )
        return tasks

    @task_template_cache(task_template_name="initialization_template")
    def create_initialization_template(self, task_template_name: str) -> TaskTemplate:
        """Stage initialization template.

        Parameters
        ----------
        task_template_name : str
            The name of the task template.

        Returns
        -------
        TaskTemplate
            The task template for stage initialization.

        """
        template = self.tool.get_task_template(
            template_name=task_template_name,
            command_template="{entrypoint} {stage_name}"
            " --experiment_dir {experiment_dir}",
            node_args=["stage_name"],
            task_args=["experiment_dir"],
            op_args=["entrypoint"],
            default_cluster_name=self.cluster_name,
        )
        if task_template_name in self.resources:
            template.set_default_compute_resources_from_dict(
                cluster_name=self.cluster_name,
                compute_resources=self.resources[task_template_name][self.cluster_name],
            )
        return template

    def create_modeling_template(self, task_template_name: str) -> TaskTemplate:
        """Stage modeling template.

        Parameters
        ----------
        task_template_name : str
            The name of the task template.

        Returns
        -------
        TaskTemplate
            The task template for stage modeling.

        """
        template = self.tool.get_task_template(
            template_name=task_template_name,
            command_template="{entrypoint}"
            " --experiment_dir {experiment_dir}"
            " --submodel_id {submodel_id}",
            node_args=["submodel_id"],
            task_args=["experiment_dir"],
            op_args=["entrypoint"],
            default_cluster_name=self.cluster_name,
        )
        if task_template_name in self.resources:
            template.set_default_compute_resources_from_dict(
                cluster_name=self.cluster_name,
                compute_resources=self.resources[task_template_name][self.cluster_name],
            )
        return template

    @task_template_cache(task_template_name="collection_template")
    def create_collection_template(self, task_template_name: str) -> TaskTemplate:
        """Stage collection template.

        Parameters
        ----------
        task_template_name : str
            The name of the task template.

        Returns
        -------
        TaskTemplate
            The task template for stage collection.

        """
        template = self.tool.get_task_template(
            template_name=task_template_name,
            command_template="{entrypoint} {stage_name}"
            " --experiment_dir {experiment_dir}",
            node_args=["stage_name"],
            task_args=["experiment_dir"],
            op_args=["entrypoint"],
            default_cluster_name=self.cluster_name,
        )
        if task_template_name in self.resources:
            template.set_default_compute_resources_from_dict(
                cluster_name=self.cluster_name,
                compute_resources=self.resources[task_template_name][self.cluster_name],
            )
        return template

    @task_template_cache(task_template_name="deletion_template")
    def create_deletion_template(self, task_template_name: str) -> TaskTemplate:
        """Stage deletion template.

        Parameters
        ----------
        task_template_name : str
            The name of the task template.

        Returns
        -------
        TaskTemplate
            The task template for stage deletion.

        """
        template = self.tool.get_task_template(
            template_name=task_template_name,
            command_template="{entrypoint} stage"
            " --experiment_dir {experiment_dir}"
            " --stage_name {stage_name}",
            node_args=["stage_name"],
            task_args=["experiment_dir"],
            op_args=["entrypoint"],
            default_cluster_name=self.cluster_name,
        )
        if task_template_name in self.resources:
            template.set_default_compute_resources_from_dict(
                cluster_name=self.cluster_name,
                compute_resources=self.resources[task_template_name][self.cluster_name],
            )
        return template

    def create_submodel_deletion_template(
        self, task_template_name: str
    ) -> TaskTemplate:
        """Stage submodel deletion template.

        Parameters
        ----------
        task_template_name : str
            The name of the task template.

        Returns
        -------
        TaskTemplate
            The task template for stage submodel deletion.

        """
        template = self.tool.get_task_template(
            template_name=task_template_name,
            command_template="{entrypoint} result --result {result}",
            node_args=["result"],
            op_args=["entrypoint"],
            default_cluster_name=self.cluster_name,
        )
        if task_template_name in self.resources:
            template.set_default_compute_resources_from_dict(
                cluster_name=self.cluster_name,
                compute_resources=self.resources[task_template_name][self.cluster_name],
            )
        return template
