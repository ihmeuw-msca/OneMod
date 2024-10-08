from onemod.constraints import bounds, no_inf
from onemod import Pipeline
from onemod.stage import PreprocessingStage, KregStage, RoverStage, SpxmodStage
from onemod.types import Data, FilePath

# Create stages
preprocessing = PreprocessingStage(
	name="1_preprocessing",
	write_to_disk=True,
	inputs=[FilePath],  # naming this "inputs" might be misleading. But because of the complexity of the "Data" model, "input_types" is also not exactly right
	outputs=[
        Data.use_validation(dict(
            col1=dict(
                type=int,
            ),
            col2=dict(
                type=float,
                constraints=[
                    bounds(ge=0),
                    no_inf()  # ???
                ]  # [0, inf)
            )
	    ))  # or onemod.types.Data.use_config(config_file=path_to_config)
	]
)

covariate_selection = RoverStage(
    name="2_covariate_selection",
    config=dict(cov_exploring=["cov1", "cov2", "cov3"]),
    groupby=["age_group_id"],
    write_to_disk=True,
    inputs=[
		Data.use_config(...)
	],
	outputs=[
		Data.use_config(...)
	]
)

global_model = SpxmodStage(
	name="3_global_model",
	write_to_disk=True,
	inputs=[
		Data.use_config(...)
	],
	outputs=[
		Data.use_config(...)
	]
)

location_model = SpxmodStage(
	name="4_location_model",
	groupby=["location_id"],
	write_to_disk=True,
	inputs=[
		Data.use_config(...)
	],
	outputs=[
		Data.use_config(...)
	]
)

smoothing = KregStage(
	name="5_smoothing",
	groupby=["region_id"],
	write_to_disk=True,
	inputs=[
		Data.use_config(...)
	],
	outputs=[
		Data.use_config(...)
	]
)

# Create pipeline
dummy_pipeline = Pipeline(
    name="dummy_pipeline",
    config=dict(
        ids=["age_group_id", "location_id", "sex_id", "year_id"],
        mtype="binomial",
    ),
    directory="/path/to/project/directory",
    data="/path/to/data.parquet",
    groupby=["sex_id"],
)

# dummy_pipeline.build() # not explicitly required as this will be a part of run()/fit()/predict(), but this generates the non-executable, intermediate representation of dummy_pipeline (and saves to JSON)
dummy_pipeline.run( # Includes validate(), build(), save() to json, <step to prepare the command for jobmon>, and go()
	tool="jobmon",
	config={
		"resources": {
			"tool_resources": {
				"slurm": {
					"queue": "all.q",
					"cores": 1,
				    "memory": "30G",
				    "runtime": "1h",
				    "project": "proj_mscm"
				}
			},
			"task_template_resources": {
				"rover_covsel_modeling_template": {
					"slurm": {
						"memory": "10G",
						"runtime": "10m"
					}
				},
				"spxmod_modeling_template": {
					"slurm": {
						"memory": "10G",
						"runtime": "20m"
					}
				},
				"weave_modeling_template": {
					"slurm": {
						"memory": "10G",
						"runtime": "10m"
					}
				},
				"ensemble_modeling_template": {
					"slurm": {
						"memory": "10G",
						"runtime": "10m"
					}
				}
			}
		},
		"subsets": {
            #...
        }
	} # and/or allow directly specifying path to config file
) # also subset definition, etc... could get complex. At this point you get a Jobmon Workflow ID, which could be referenced to resume, etc. But the non-jobmon-specific dummy_pipeline representation still exists.