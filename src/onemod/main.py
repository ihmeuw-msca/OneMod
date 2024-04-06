"""Run onemod pipeline."""

from __future__ import annotations

from pathlib import Path

import fire
from jobmon.client.status_commands import resume_workflow_from_id

from onemod.scheduler.scheduler import Scheduler
from onemod.schema.validate import validate_config
from onemod.utils import as_list, get_handle


def run_pipeline(
    directory: str,
    stages: str | list[str] | None = None,
    cluster_name: str = "slurm",
    configure_resources: bool = True,
    run_local: bool = False,
) -> None:
    """Run onemod pipeline.

    Parameters
    ----------
    directory : str
        The experiment directory. It must contain config/settings.yml.
    stages : list of str or str, optional
        The pipeline stages to run. Default is ['rover', 'weave', 'ensemble'].
    save_intermediate : bool, optional
        Whether to save intermediate stage results. Default is False.
    cluster_name : str, optional
        Name of the cluster to run the pipeline on. Default is 'slurm'.
    configure_resources : bool, optional
        Whether to configure resources in directory/config/resources.yml. Default is True.

    """
    all_stages = ["rover_covsel", "regmod_smooth", "weave", "ensemble"]
    if stages is None:
        stages = all_stages
    for stage in as_list(stages):
        if stage not in all_stages:
            raise ValueError(f"Invalid stage: {stage}")

    # Load and validate the configuration file
    dataif, config = get_handle(directory)
    validate_config(stages=stages, directory=directory, config=config)

    directory = Path(directory)

    if configure_resources and not run_local:
        resources_file = str(directory / "config" / "resources.yml")
    else:
        resources_file = ""

    # Create the scheduler and run it
    scheduler = Scheduler(
        stages=stages,
        directory=directory,
        config=config,
        resources_path=resources_file,
        default_cluster_name=cluster_name,
        configure_resources=configure_resources,
    )

    scheduler.run(run_local=run_local)


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
