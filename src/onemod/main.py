"""Run onemod pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import fire
from jobmon.client.api import Tool

from onemod.stage import StageTemplate
from onemod.utils import as_list

try:
    # Import will fail until the next version of jobmon is released;
    # eventually client.api will become the desired import source
    from jobmon.client.api import resume_workflow_from_id  # type: ignore
except ImportError:
    from jobmon.client.status_commands import resume_workflow_from_id

if TYPE_CHECKING:
    from jobmon.client.workflow import Workflow


upstream_dict: dict[str, list] = {
    "rover": [],
    "swimr": ["rover"],
    "weave": ["rover"],
    "ensemble": ["swimr", "weave"],
}


def create_workflow(
    directory: str,
    stages: list[str],
    save_intermediate: bool,
    cluster_name: str,
    configure_resources: bool,
    tool: Optional[Tool] = None,
) -> Workflow:
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
        The jobmon Tool instance to use for creating the workflow. If not provided, a new Tool instance
        named "onemod_tool" will be created with default compute resources set.

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

    # Create stage tasks
    for stage, upstream_stages in upstream_dict.items():
        if stage in stages:
            stage_template = StageTemplate(
                stage_name=stage,
                experiment_dir=experiment_dir,
                save_intermediate=save_intermediate,
                cluster_name=cluster_name,
                resources_file=resources_file,
                tool=tool,
            )
            upstream_tasks = []
            for upstream_stage in upstream_stages:
                if upstream_stage in stages:
                    upstream_task = workflow.get_tasks_by_node_args(
                        "collection_template", stage_name=upstream_stage
                    )
                    upstream_tasks.append(upstream_task[0])
            workflow.add_tasks(
                stage_template.create_tasks(upstream_tasks=upstream_tasks)
            )
    return workflow


def run_pipeline(
    directory: str,
    stages: Optional[Union[list[str], str]] = None,
    save_intermediate: bool = False,
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
    if stages is None:
        stages = ["rover", "swimr", "weave", "ensemble"]
    for stage in as_list(stages):
        if stage not in ["rover", "swimr", "weave", "ensemble"]:
            raise ValueError(f"Invalid stage: {stage}")
    workflow = create_workflow(
        directory=directory,
        stages=as_list(stages),
        save_intermediate=save_intermediate,
        cluster_name=cluster_name,
        configure_resources=configure_resources,
    )
    status = workflow.run(configure_logging=True, seconds_until_timeout=24 * 60 * 60)
    if status != "D":
        raise ValueError(f"workflow {workflow.name} failed: {status}")


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

    This function is intended to be executed using the 'fire.Fire' method to enable command-line
    execution of the 'run_pipeline' and 'resume_pipeline' functions.

    """
    fire.Fire(
        {
            "run_pipeline": run_pipeline,
            "resume_pipeline": resume_pipeline,
        }
    )
