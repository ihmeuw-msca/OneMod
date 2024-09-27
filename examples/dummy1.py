"""Dummy example."""

from onemod import Pipeline
from onemod.stage.data_stage import PreprocessingStage
from onemod.stage.model_stage import RoverStage, SpxmodStage, KregStage

# Adding TypeHints

preprocessing = PreprocessingStage(
    # Input: FilePath
    inputs = [ onemod.types.filepath],
    name="1_preprocessing", config=dict(data="/path/to/data.parquet")
    # ToDo What can be stated about the data:
    # Static Dimensionality, for each colum
    #  -Name of col
    #  -Data type of column
    # Bound constraints:
    # e.g 0..1
    # strictly positive
    # 0..10 billion for populations

  outputs = [onemod.types.data{
    "age_group_id": {
        type = onemod.types.integer,
        constrint = onemod.types.bound(0, 100),
    },
    "location_id": onemod.types.integer,
    onemod.types.some_other_stuff  # special keyword for ....
  }],
)

covariate_selection = RoverStage(
inputs = [onemod.types.data{
    "age_group_id": {
        type = onemod.types.integer,
        constrint = onemod.types.bound(0, 100),
    },
    "location_id": onemod.types.integer,
    onemod.types.some_other_stuff  # special keyword for ....
  }

    name="2_covariate_selection",
    config=dict(cov_exploring=["cov1", "cov2", "cov3"]),
    groupby=["age_group_id"],
)

global_model = SpxmodStage(name="3_global_model")

location_model = SpxmodStage(name="4_location_model", groupby=["location_id"])

smoothing = KregStage(name="5_smoothing", groupby=["region_id"])

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

dummy_pipeline.add_stages(
    [
        preprocessing,
        covariate_selection,
        global_model,
        location_model,
        smoothing,
    ]
)

data = preprocessing()
selected_covs = covariate_selection(data=data)
global_offset = global_model(data=data, selected_covs=selected_covs)
location_offset = location_model(data=data, offset=global_offset)
predictions = smoothing(data=data, offset=location_offset)

dummy_pipeline.compile()
  # This call walks through the pipeline from start to finish and checks that the current stage produces
  # sufficient outputs for the immediate downstream stages.
  # Bonus points: check that any range-constraints or bound constraints are compatible.
  # Result of the compile is either a serialsisd pipeline or a set of error messages.
dummy_pipeline.run()
