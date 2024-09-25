"""Dummy example."""

from onemod import Pipeline
from onemod.stage.data_stage import PreprocessingStage
from onemod.stage.model_stage import RoverStage, SpxmodStage, KregStage

preprocessing = PreprocessingStage(
    name="1_preprocessing", config=dict(data="/path/to/data.parquet")
)

covariate_selection = RoverStage(
    name="2_covariate_selection",
    config=dict(
        data=preprocessing.data, cov_exploring=["cov1", "cov2", "cov3"]
    ),
    groupby=["age_group_id"],
)

global_model = SpxmodStage(
    name="3_global_model",
    config=dict(
        data=preprocessing.data, selected_covs=covariate_selection.selected_covs
    ),
)

location_model = SpxmodStage(
    name="4_location_model",
    config=dict(data=preprocessing.data, offset=global_model.predictions),
    groupby=["location_id"],
)

smoothing = KregStage(
    name="5_smoothing",
    config=dict(data=preprocessing.data, offset=location_model.predictions),
    groupby=["region_id"],
)

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

dummy_pipeline.compile()
dummy_pipeline.run()
