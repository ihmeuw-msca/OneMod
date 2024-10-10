"""Example OneMod pipeline."""

import fire

from onemod import Pipeline
from onemod.stage import PreprocessingStage, RoverStage, SpxmodStage, KregStage

from custom_stage import CustomStage


def create_pipeline(directory: str, data: str):
    # Create stages
    # Where should stages and/or stage instances define input requirements?
    preprocessing = PreprocessingStage(name="preprocessing", config={})
    covariate_selection = RoverStage(
        name="covariate_selection",
        config={"cov_exploring": ["cov1", "cov2", "cov3"]},
        groupby=["age_group_id"],
    )
    global_model = SpxmodStage(name="global_model", config={})
    location_model = SpxmodStage(
        name="location_model", config={}, groupby=["location_id"]
    )
    smoothing = KregStage(name="smoothing", config={}, groupby=["region_id"])
    custom_stage = CustomStage(
        name="custom_stage",
        config={"custom_param": [1, 2]},
        groupby=["super_region_id"],
    )

    # Create pipeline
    example_pipeline = Pipeline(
        name="example_pipeline",
        config={
            "ids": ["age_group_id", "location_id", "sex_id", "year_id"],
            "mtype": "binomial",
        },
        directory=directory,
        data=data,
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
            custom_stage,
        ]
    )

    # Define dependencies
    # Is this where data validation (where possible) should happen? Or in compile?
    preprocessing(data=example_pipeline.data)
    covariate_selection(data=preprocessing.output["data"])
    global_model(
        data=preprocessing.output["data"],
        selected_covs=covariate_selection.output["selected_covs"],
    )
    location_model(
        data=preprocessing.output["data"],
        offset=global_model.output["predictions"],
    )
    smoothing(
        data=preprocessing.output["data"],
        offset=location_model.output["predictions"],
    )
    custom_stage(
        observations=preprocessing.output["data"],
        predictions=smoothing.output["predictions"],
    )

    # Save pipeline config
    example_pipeline.to_json()


if __name__ == "__main__":
    fire.Fire(create_pipeline)


# Compile pipeline
# - Validate DAG
# - Validate data?
# - Pass pipeline directory and config to stages
# - Create stage subsets and parameter sets
# - Save pipeline JSON
# example_pipeline.compile()

# Run (fit and predict) entire pipeline
# example_pipeline.run()

# Fit some stages
# example_pipeline.fit(stages=["preprocessing", "covariate_selection"])

# Predict for some locations
# What's the best syntax for this?
# example_pipeline.predict(id_subsets={"location_id": [1, 2, 3]})

# Run pipeline the command line
# onemod --config /path/to/pipeline/config.json

# Fit pipeline with jobmon from the command line
# onemod --config /path/to/pipeline/config.json --method fit --backend jobmon --cluster cluster_name --resources /path/to/resources.yaml

# Predict stage from the command line
# onemod --config /path/to/pipeline/config.json --stage_name stage_name --from_pipeline --method predict
# onemod --config /path/to/stage/config.json --method predict
