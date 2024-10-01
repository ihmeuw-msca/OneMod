"""Define dependencies before adding stages."""

from onemod import Pipeline
from onemod.stage.data_stages import PreprocessingStage
from onemod.stage.model_stages import KregStage, RoverStage, SpxmodStage

# Create stages
preprocessing = PreprocessingStage(
    name="1_preprocessing",
    config=dict(),
    input=dict(data="/path/to/data.parquet"),
)

covariate_selection = RoverStage(
    name="2_covariate_selection",
    config=dict(cov_exploring=["cov1", "cov2", "cov3"]),
    input=dict(data=preprocessing.data),
    groupby=["age_group_id"],
)

global_model = SpxmodStage(
    name="3_global_model",
    config=dict(),
    input=dict(
        data=preprocessing.data, selected_covs=covariate_selection.selected_covs
    ),
)

location_model = SpxmodStage(
    name="4_location_model",
    config=dict(),
    input=dict(data=preprocessing.data, offset=global_model.predictions),
    groupby=["location_id"],
)

smoothing = KregStage(
    name="5_smoothing",
    config=dict(),
    input=dict(data=preprocessing.data, offset=location_model.predictions),
    groupby=["region_id"],
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

# Run pipeline
dummy_pipeline.compile()
dummy_pipeline.run()
