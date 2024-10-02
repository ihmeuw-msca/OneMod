"""Define dependencies after adding stages."""

from onemod import Pipeline
from onemod.stage.data_stages import PreprocessingStage
from onemod.stage.model_stages import KregStage, RoverStage, SpxmodStage

# Create stages
preprocessing = PreprocessingStage(name="1_preprocessing")

covariate_selection = RoverStage(
    name="2_covariate_selection",
    config=dict(cov_exploring=["cov1", "cov2", "cov3"]),
    groupby=["age_group_id"],
)

global_model = SpxmodStage(name="3_global_model")

location_model = SpxmodStage(name="4_location_model", groupby=["location_id"])

smoothing = KregStage(name="5_smoothing", groupby=["region_id"])

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

# Add stages
dummy_pipeline.add_stages(
    [
        preprocessing,
        covariate_selection,
        global_model,
        location_model,
        smoothing,
    ]
)

# Define dependencies
preprocessing(data="/path/to/input/data.parquet")
covariate_selection(data=preprocessing.output["data"])
global_model(
    data=preprocessing.output["data"],
    selected_covs=covariate_selection.output["selected_covs"],
)
location_model(
    data=preprocessing.output["data"], offset=global_model.output["predictions"]
)
predictions = smoothing(
    data=preprocessing.output["data"],
    offset=location_model.output["predictions"],
)

# Run pipeline
dummy_pipeline.compile()
dummy_pipeline.run()
