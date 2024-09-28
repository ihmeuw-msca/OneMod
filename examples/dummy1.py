"""Dummy example."""

from onemod import Pipeline
from onemod.stage.data_stages import PreprocessingStage
from onemod.stage.model_stages import RoverStage, SpxmodStage, KregStage

preprocessing = PreprocessingStage(
    name="1_preprocessing", config=dict(data="/path/to/data.parquet")
)

covariate_selection = RoverStage(
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
dummy_pipeline.run()
