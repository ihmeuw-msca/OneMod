"""Run onemod pipeline."""
from __future__ import annotations

from pathlib import Path
from pplkit.data.interface import DataInterface
import shutil
from typing import Optional, TYPE_CHECKING, Union

import fire
from jobmon.client.api import Tool
from pydantic import ValidationError

from onemod.schema.config import ParentConfiguration
from onemod.schema.validate import validate_config
from onemod.orchestration.stage import StageTemplate
from onemod.orchestration.templates import create_initialization_template
from onemod.utils import as_list, get_data_interface

try:
    # Import will fail until the next version of jobmon is released;
    # eventually client.api will become the desired import source
    from jobmon.client.api import resume_workflow_from_id  # type: ignore
except ImportError:
    from jobmon.client.status_commands import resume_workflow_from_id

if TYPE_CHECKING:
    from jobmon.client.workflow import Workflow


upstream_dict: dict[str, list] = {
    "rover_covsel": [],
    "regmod_smooth": ["rover_covsel"],
    "swimr": ["regmod_smooth"],
    "weave": ["regmod_smooth"],
    "ensemble": ["swimr", "weave"],
}


def create_workflow(
    directory: str,
    config: ParentConfiguration,
    stages: list[str],
    save_intermediate: bool,
    cluster_name: str,
    configure_resources: bool,
    tool: Optional[Tool] = None,
) -> "Workflow":
    """Create OneMod workflow.

    Parameters
    ----------
    directory : str
        The experiment directory. It must contain config/settings.yml.
    stages : list of str
        The pipeline stages to run.
    save_intermediate : bool
        Whether to save intermediate stage results.
    cluster_name : str
        Name of the cluster to run the pipeline on.
    configure_resources : bool
        Whether to configure resources in directory/config/resources.yml.
    tool : Tool, optional
        The jobmon Tool instance to use for creating the workflow. If not provided, a new
        Tool instance named "onemod_tool" will be created with default compute resources set.

    Returns
    -------
    Workflow
        The created OneMod workflow.

    """
    experiment_dir = Path(directory)
    if configure_resources:
        resources_file = str(experiment_dir / "config" / "resources.yml")
    else:
        resources_file = ""

    # Jobmon setup
    if tool is None:
        tool = Tool(name="onemod_tool")
        if configure_resources:
            tool.set_default_compute_resources_from_yaml(
                default_cluster_name=cluster_name,
                yaml_file=resources_file,
            )
    workflow = tool.create_workflow(name="onemod_workflow")
    workflow.set_default_cluster_name(cluster_name=cluster_name)
    workflow.set_default_max_attempts(value=1)

    # Create the initialization task.
    initialization_template = create_initialization_template(
        tool=tool,
        resources_file=resources_file,
    )

    initialization_task = initialization_template.create_task(
        experiment_dir=experiment_dir,
        # A quirk of Fire is that lists have to be in space-less comma separated format.
        stages=f"[{','.join(stages)}]",
        entrypoint=shutil.which("initialize_results"),
    )

    workflow.add_task(initialization_task)

    # Create stage tasks
    for stage, upstream_stages in upstream_dict.items():
        if stage in stages:
            stage_template = StageTemplate(
                stage_name=stage,
                config=config,
                experiment_dir=experiment_dir,
                save_intermediate=save_intermediate,
                cluster_name=cluster_name,
                resources_file=resources_file,
                tool=tool,
            )
            # Technically only the first stages needs to depend on the initialization task,
            # but it's probably cleaner in code to make it a dependency for all stages.
            upstream_tasks = [initialization_task]
            for upstream_stage in upstream_stages:
                if upstream_stage in stages:
                    upstream_task = workflow.get_tasks_by_node_args(
                        "collection_template", stage_name=upstream_stage
                    )
                    upstream_tasks.extend(upstream_task)
            stage_tasks = stage_template.create_tasks(upstream_tasks=upstream_tasks)
            workflow.add_tasks(stage_tasks)
    return workflow


def run_pipeline(
    directory: str,
    stages: Optional[Union[list[str], str]] = None,
    save_intermediate: bool = True,
    cluster_name: str = "slurm",
    configure_resources: bool = True,
) -> None:
    """Run onemod pipeline.

    Parameters
    ----------
    directory : str
        The experiment directory. It must contain config/settings.yml.
    stages : list of str or str, optional
        The pipeline stages to run. Default is ['rover', 'swimr', 'weave', 'ensemble'].
    save_intermediate : bool, optional
        Whether to save intermediate stage results. Default is False.
    cluster_name : str, optional
        Name of the cluster to run the pipeline on. Default is 'slurm'.
    configure_resources : bool, optional
        Whether to configure resources in directory/config/resources.yml. Default is True.

    """
    all_stages = ["rover_covsel", "regmod_smooth", "swimr", "weave", "ensemble"]
    if stages is None:
        stages = all_stages
    for stage in as_list(stages):
        if stage not in all_stages:
            raise ValueError(f"Invalid stage: {stage}")

    # Validate the configuration file
    dataif = get_data_interface(directory)
    config = _load_validated_config(dataif=dataif, stages=stages, experiment_dir=directory)

    workflow = create_workflow(
        directory=directory,
        config=config,
        stages=as_list(stages),
        save_intermediate=save_intermediate,
        cluster_name=cluster_name,
        configure_resources=configure_resources,
    )
    status = workflow.run(configure_logging=True, seconds_until_timeout=24 * 60 * 60)
    if status != "D":
        raise ValueError(f"workflow {workflow.name} failed: {status}")


def _load_validated_config(dataif: DataInterface, stages: list[str], experiment_dir: str):
    settings = ParentConfiguration(**dataif.load_settings())
    validate_config(
        stages=stages,
        directory=experiment_dir,
        config=settings,
    )
    return settings


def resume_pipeline(workflow_id: int, cluster_name: str = "slurm") -> None:
    """Resume onemod pipeline from last point of failure.

    Parameters
    ----------
    workflow_id : int
        ID of the workflow to resume.
    cluster_name : str, optional
        Name of cluster to run pipeline on. Default is 'slurm'.

    """
    resume_workflow_from_id(
        workflow_id=workflow_id, cluster_name=cluster_name, log=True
    )


def main() -> None:
    """Entry point for running onemod pipeline using command-line interface.

    This function is intended to be executed using the 'fire.Fire' method to enable
    command-line execution of the 'run_pipeline' and 'resume_pipeline' functions.

    """
    fire.Fire(
        {
            "run_pipeline": run_pipeline,
            "resume_pipeline": resume_pipeline,
        }
    )
