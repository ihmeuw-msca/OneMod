"""Run OneMod pipeline using Jobmon."""

import sys

import fire
from jobmon.client.api import Tool

from onemod import Pipeline, get_task_template
from onemod.stage import GroupedStage, CrossedStage


def run_pipeline(
    cluster_name: str,
    resources_yaml: str,
    config_json: str,
    method: str = "run",
) -> None:
    # Load pipeline
    example_pipeline = Pipeline.from_json(filepath=config_json)

    # Create tool
    tool = Tool(name="example_tool")
    tool.set_default_cluster_name(cluster_name)
    tool.set_default_compute_resources_from_yaml(
        cluster_name, resources_yaml, set_task_templates=True
    )

    # Create tasks
    tasks = []
    upstream_tasks = []
    args = {
        "python": sys.executable,  # potential to run tasks on different envs
        "filepath": config_json,
    }
    for stage in example_pipeline.stages.values():
        if method not in stage._skip_if:
            node_args = {}
            if isinstance(stage, GroupedStage) and stage.subset_ids:
                node_args["subset_id"] = stage.subset_ids
            if isinstance(stage, CrossedStage) and stage.param_ids:
                node_args["param_id"] = stage.param_ids

            task_template = get_task_template(
                tool=tool,
                stage_name=stage.name,
                method=method,
                subsets="subset_id" in node_args,
                params="param_id" in node_args,
            )

            if "subset_id" in node_args or "param_id" in node_args:
                upstream_tasks = task_template.create_tasks(
                    name=f"{stage.name}_{method}_task",
                    upstream_tasks=upstream_tasks,
                    max_attempts=1,
                    **{**args, **node_args},
                )
                tasks.extend(upstream_tasks)

                upstream_tasks = [
                    get_task_template(
                        tool=tool, stage_name=stage.name, method="collect"
                    ).create_task(
                        name=f"{stage.name}_collect_task",
                        upstream_tasks=upstream_tasks,
                        max_attempts=1,
                        **args,
                    )
                ]
                tasks.extend(upstream_tasks)
            else:
                upstream_tasks = [
                    task_template.create_task(
                        name=f"{stage.name}_{method}_task",
                        upstream_tasks=upstream_tasks,
                        max_attempts=1,
                        **args,
                    )
                ]
                tasks.extend(upstream_tasks)

    # Create workflow
    workflow = tool.create_workflow(name="example_workflow")
    workflow.add_tasks(tasks)
    workflow.bind()
    print(f"workflow_id: {workflow.workflow_id}")
    status = workflow.run()
    if status != "D":
        raise ValueError("Workflow {workflow.workflow_id} failed")


if __name__ == "__main__":
    fire.Fire(run_pipeline)
