"""Create Jobmon workflow using Pipeline and Stage objects."""

import sys

from jobmon.client.api import Tool

from onemod import Pipeline, get_task_template
from onemod.stage import (
    GroupedStage,
    CrossedStage,
    PreprocessingStage,
    KregStage,
    RoverStage,
    SpxmodStage,
)

from custom_stage import CustomStage

DIRECTORY = "/path/to/directory"
DATA = "/path/to/data.parquet"
METHOD = "run"

# Create stages
preprocessing = PreprocessingStage(name="preprocessing", config=dict())
covariate_selection = RoverStage(
    name="covariate_selection",
    config=dict(cov_exploring=["cov1", "cov2", "cov3"]),
    groupby=["age_group_id"],
)
global_model = SpxmodStage(name="global_model")
location_model = SpxmodStage(name="location_model", groupby=["location_id"])
smoothing = KregStage(name="smoothing", groupby=["region_id"])
plotting = CustomStage(name="plotting")

# Create pipeline
example_pipeline = Pipeline(
    name="example_pipeline",
    config=dict(
        ids=["age_group_id", "location_id", "sex_id", "year_id"],
        mtype="binomial",
    ),
    directory=DIRECTORY,
    data=DATA,
    groupby=["sex_id"],
)

# Add stages
example_pipeline.add_stages(
    [
        preprocessing,
        covariate_selection,
        global_model,
        location_model,
        smoothing,
    ]
)

# Define dependencies
preprocessing(data=DATA)
covariate_selection(data=preprocessing.output["data"])
global_model(
    data=preprocessing.output["data"],
    selected_covs=covariate_selection.output["selected_covs"],
)
location_model(
    data=preprocessing.output["data"], offset=global_model.output["predictions"]
)
smoothing(
    data=preprocessing.output["data"],
    offset=location_model.output["predictions"],
)

# Save pipeline config
example_pipeline.to_json()

# Create tool
tool = Tool(name="example_tool")
tool.set_default_cluster_name("slurm")
tool.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    compute_resources={
        "queue": "all.q",
        "cores": 1,
        "memory": "1G",
        "runtime": "00:01:00",
        "project": "proj_mscm",
    },
)

# Create tasks
tasks = []
upstream_tasks = []
args = {
    "python": sys.executable,  # potential to run tasks on different envs
    "filepath": example_pipeline.directory / (example_pipeline.name + ".json"),
}
for stage in example_pipeline.stages.values():
    if METHOD not in stage._skip_if:
        node_args = {}
        if isinstance(stage, GroupedStage) and stage.subset_ids:
            node_args["subset_id"] = stage.subset_ids
        if isinstance(stage, CrossedStage) and stage.param_ids:
            node_args["param_id"] = stage.param_ids

        task_template = get_task_template(
            tool=tool,
            stage_name=stage.name,
            method=METHOD,
            subsets="subset_id" in node_args,
            params="param_id" in node_args,
        )

        if "subset_id" in node_args or "param_id" in node_args:
            upstream_tasks = task_template.create_tasks(
                name=f"{stage.name}_{METHOD}_task",
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
                    name=f"{stage.name}_{METHOD}_task",
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
    raise ValueError("Workflow failed")
