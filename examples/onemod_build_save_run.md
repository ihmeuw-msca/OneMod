# OneMod Pipeline Construction

## Terminology and Notes

### Configuration

At a high level, there are two classes of configuration:

  1. Pipeline configuration, known as "settings" in OneMod 1.0
    - Examples:
      - Project directory
      - ID vars (e.g. ["age_group_id", "sex_id", "location_id"], etc.)
      - ID subsets (dict) - keys must be in ID vars list or set
      - holdouts
      - Stage-level configurations
        - groupby
        - arbitarily many others, as defined by the particular stage. May be nested (ex: spxmod config)
  2. Execution configuration (partly represented by "resources.yaml" in OneMod 1.0)
    - Examples:
      - Cluster queue
      - Requested resources
      - Default/max runtime
      - Other Jobmon-specific parameters

As you might expect, the former are specified in the Pipeline construction (or via config file(s)) as arguments passed at the time of instantiation of your Stage and Pipeline classes, and are at the Pipeline level. These options are a part of what gets saved as the Pipeline representation prior to runtime

The latter are specified in the run() (or fit() or predict() call), and are at the run/workflow level. These are not saved with the Pipeline itself, and represent only configurations that are required for execution. These would vary based on user runtime environment and choice of workflow orchestration tool, for instance.

## How to set up a OneMod Pipeline with Stages

Specify all Stage-specific and Pipeline instance-specifc configurations via Stages and Pipeline instantiation, e.g.:

```python
from onemod.constraints import bounds, no_inf
from onemod.types import FilePath, Integer, Float

# Create stages
preprocessing = PreprocessingStage(
    name="1_preprocessing",
    write_to_disk=True,
    inputs=[FilePath],  # naming this "inputs" might be misleading. But because of the complexity of the "Data" model, "input_types" is also not exactly right
    outputs=[
        Data.use_config(
            {
                "col1": {
                    type: Integer,  # Note - type is a reserved word. Problem?
                },
                "col2": {
                    type: Float,
                    constraints: [
                        bounds(ge=0),
                        no_inf()  # ???
                    ]  # [0, inf)
                }
            }
        )  # or onemod.types.Data.use_config(config_file=path_to_config)
    ]
)

covariate_selection = RoverStage(
    name="2_covariate_selection",
    config=dict(cov_exploring=["cov1", "cov2", "cov3"]),
    groupby=["age_group_id"],
    write_to_disk=True,
    inputs=[...],  # Similar structure to above. From this point on, some instance of `Data` will be most common input/output class
    outputs=[...],
)

global_model = SpxmodStage(
    name="3_global_model",
    write_to_disk=True,
    inputs=[...],
    outputs=[...]   
)

location_model = SpxmodStage(
    name="4_location_model",
    groupby=["location_id"],
    write_to_disk=True,
    inputs=[...],
    outputs=[...]
)

smoothing = KregStage(
    name="5_smoothing",
    groupby=["region_id"],
    write_to_disk=True,
    inputs=[...],
    outputs=[...]
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
```

## How to run a OneMod Pipeline

In these examples, the orchestration tool, etc. is not tied to my_pipeline, but to the execution. Orchestration tool/execute strategy must be specified with whichever execute method call, and this approach puts more responsibility into run(), fit(), predict().

Example 1:

```python
# Define Pipeline, stages, data flow...
# ...

# my_pipeline.build() # not explicitly required as this will be a part of run()/fit()/predict(), but this generates the non-executable, intermediate representation of my_pipeline (and saves to JSON)
my_pipeline.run( # Includes validate(), build(), save() to json, <step to prepare the command for jobmon>, and go()
 tool="jobmon",
 config=dict({
  "resources": {
   "tool_resources": {
    "slurm": {
     "queue": "all.q",
     "cores": 1
        "memory": "30G"
        "runtime": "1h"
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
  subsets: {
   ...
  }
 }) # and/or allow directly specifying path to config file
) # also subset definition, etc... could get complex. At this point you get a Jobmon Workflow ID, which could be referenced to resume, etc. But the non-jobmon-specific my_pipeline representation still exists.
```

Example 2:

```python
# Define Pipeline, stages, data flow...
# ...

my_pipeline.fit(
    tool="jobmon",
    config=dict({
        "resources": ...
        "subsets": {
            ...
        }
    })
)
```

Example 3:

```python
# Define Pipeline, stages, data flow...
# ...
my_pipeline.fit(
    tool="jobmon",
    config=dict({
        "resources": ...
    })
)
# ...then maybe after some inspection, etc.
my_pipeline.predict(
    tool="sequential",
    config=dict({
        ...
    })
)
```

It is not common that you would specify a different tool for each, but the point is that it is possible, i.e. each call to run()/fit()/predict() represents an isolated execution.

---

## Usage Summary

- Model-related parameters are passed directly via the configs for Stages and Pipeline(), so we don’t need to worry about passing these to build() or anything after build()
- Users should specify execution options within their run|fit|predict() calls, most often probably by pointing to config file(s)
- build() method may be run standalone, but because the “run, make changes to params/config, run, make changes to params/config, repeat...” model development loop is so common, run/fit/predict start by validating and building the pipeline (and so of course you still get the benefit of build-time validation errors caught early any time you run)
- If someone is iteratively testing out different model configurations, after updating their Pipeline object the command they need to re-execute is simply run|fit|predict(), as validating and building is included by default
