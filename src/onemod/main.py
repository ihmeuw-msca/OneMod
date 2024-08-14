"""Run onemod pipeline."""

from __future__ import annotations

from pathlib import Path

import fire

try:
    from jobmon.client.status_commands import resume_workflow_from_id
except ImportError:
    pass


from onemod.scheduler.scheduler import Scheduler
from onemod.scheduler.scheduling_utils import SchedulerType
from onemod.utils import format_input, get_handle


def run_pipeline(
    directory: str,
    stages: list[str] | None = None,
    cluster_name: str = "slurm",
    run_local: bool = False,
    jobmon: bool = False,
) -> None:
    """Run onemod pipeline.

    Parameters
    ----------
    directory : str
        The experiment directory. It must contain config/settings.yml.
    stages : list of str, optional
        The pipeline stages to run. Default is ['rover_covsel', 'spxmod', 'weave', 'ensemble'].
    cluster_name : str, optional
        Name of the cluster to run the pipeline on. Default is 'slurm'.
        For testing, use 'dummy'. Otherwise the directory must contain
        config/resources.yml.
    run_local : bool, optional
        If true run the jobs sequentially without Jobmon. Default is False.
    jobmon : bool, optional
        If True use Jobmon. Default is True.

    """
    if run_local and jobmon:
        raise ValueError("Exactly one of run_local and jobmon can be True")

    # If both false then use jobmon because it means they did not specify anything on the command line
    if not run_local and not jobmon:
        jobmon = True

    scheduler_type: SchedulerType = (
        SchedulerType.jobmon if jobmon else SchedulerType.run_local
    )
    _run_pipeline(directory, stages, cluster_name, scheduler_type)


def _run_pipeline(
    directory: str,
    stages: list[str] | None = None,
    cluster_name: str = "slurm",
    scheduler_type: SchedulerType = SchedulerType.jobmon,
) -> None:
    """
    Internal function that uses an enum for the scheduler type for clarity.
    Fire cannot handle enums.
    """

    all_stages = ["rover_covsel", "spxmod", "weave", "ensemble"]
    if stages is None:
        stages = all_stages
    for stage in stages:
        if stage not in all_stages:
            raise ValueError(f"Invalid stage: {stage}")

    # Load and validate the configuration file
    dataif, config = get_handle(directory)

    # Filter input data by ID subsets
    # Used for task creation so must happen outside of workflow
    if "rover_covsel" in stages or not dataif.data.exists():
        format_input(directory)

    # Configure Jobmon resources
    directory = Path(directory)
    if scheduler_type == SchedulerType.jobmon and cluster_name != "dummy":
        resources_yaml = str(directory / "config" / "resources.yml")
    else:
        resources_yaml = ""

    # Create the scheduler and run it
    scheduler = Scheduler(
        directory=directory,
        config=config,
        stages=stages,
        default_cluster_name=cluster_name,
        resources_yaml=resources_yaml,
    )
    scheduler.run(scheduler_type=scheduler_type)


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
    # Only expose the run_pipeline and resume_pipeline functions
    fire.Fire(
        {"run_pipeline": run_pipeline, "resume_pipeline": resume_pipeline}
    )
