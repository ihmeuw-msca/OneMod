"""Define dependencies after adding stages."""

from onemod import Pipeline
from onemod.stage import PreprocessingStage, KregStage, RoverStage, SpxmodStage

# Create stages
# Where should stages and/or stage instances define input requirements?
preprocessing = PreprocessingStage(name="1_preprocessing")

covariate_selection = RoverStage(
    name="2_covariate_selection",
    config={"cov_exploring": ["cov1", "cov2", "cov3"]},
    groupby=["age_group_id"],
)

global_model = SpxmodStage(name="3_global_model", config={})

location_model = SpxmodStage(
    name="4_location_model", config={}, groupby=["location_id"]
)

smoothing = KregStage(name="5_smoothing", config={}, groupby=["region_id"])

# Create pipeline
dummy_pipeline = Pipeline(
    name="dummy_pipeline",
    config={
        "ids": ["age_group_id", "location_id", "sex_id", "year_id"],
        "mtype": "binomial",
    },
    directory="/path/to/project/directory",
    data="/path/to/input/data.parquet",
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
# Is this where data validation (where possible) should happen? Or in compile?
preprocessing(data=dummy_pipeline.data)
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

# Compile pipeline
# - Validate DAG
# - Validate data?
# - Pass pipeline directory and config to stages
# - Create stage subsets and parameter sets
# - Save pipeline JSON
dummy_pipeline.compile()

# Run (fit and predict) entire pipeline
dummy_pipeline.run()

# Fit some stages
dummy_pipeline.fit(stages=["preprocessing", "covariate_selection"])

# Predict for some locations
# What's the best syntax for this?
dummy_pipeline.predict(id_subsets={"location_id": [1, 2, 3]})
