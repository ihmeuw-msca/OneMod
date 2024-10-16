"""Example OneMod pipeline."""

import fire

from onemod import Pipeline
from onemod.stage import PreprocessingStage, RoverStage, SpxmodStage, KregStage

from custom_stage import CustomStage


def create_pipeline(directory: str, data: str):
    # Create stages
    # TODO: Does stage-specific validation info go here or in class definitions?
    preprocessing = PreprocessingStage(name="preprocessing", config={})
    covariate_selection = RoverStage(
        name="covariate_selection",
        config={"cov_exploring": ["cov1", "cov2", "cov3"]},
        groupby=["age_group_id"],
    )
    global_model = SpxmodStage(
        name="global_model",
        config={"xmodel": {"variables": [{"name": "var1"}, {"name": "var2"}]}},
    )
    location_model = SpxmodStage(
        name="location_model",
        config={"xmodel": {"variables": [{"name": "var1"}, {"name": "var2"}]}},
        groupby=["location_id"],
    )
    smoothing = KregStage(
        name="smoothing",
        config={
            "kreg_model": {
                "age_scale": 1,
                "gamma_age": 1,
                "gamma_year": 1,
                "exp_location": 1,
                "lam": 1,
                "nugget": 1,
            },
            "kreg_fit": {
                "gtol": 1,
                "max_iter": 1,
                "cg_maxiter": 1,
                "cg_maxiter_increment": 1,
                "nystroem_rank": 1,
            },
        },
        groupby=["region_id"],
    )
    custom_stage = CustomStage(
        name="custom_stage",
        config={"custom_param": [1, 2]},
        groupby=["super_region_id"],
    )

    # Create pipeline
    example_pipeline = Pipeline(
        name="example_pipeline",
        config={
            "id_columns": ["age_group_id", "location_id", "sex_id", "year_id"],
            "model_type": "binomial",
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

    # Serialize pipeline
    example_pipeline.to_json()

    # TODO: Validate and serialize
    # User could call this method themself, but run/fit/predict should
    # probably also call it in case updates have been made to the
    # pipeline (e.g., someone is experimenting with a pipeline in a
    # a notebook)
    # example_pipeline.build()

    # Run (fit and predict) entire pipeline
    # example_pipeline.run()

    # TODO: Fit specific stages
    # example_pipeline.fit(stages=["preprocessing", "covariate_selection"])

    # TODO: Predict for specific locations
    # example_pipeline.predict(id_subsets={"location_id": [1, 2, 3]})


if __name__ == "__main__":
    fire.Fire(create_pipeline)


# Run pipeline the command line
# onemod --config /path/to/pipeline/config.json

# Fit pipeline with jobmon from the command line
# onemod --config /path/to/pipeline/config.json --method fit --backend jobmon --cluster cluster_name --resources /path/to/resources.yaml

# Predict stage from the command line using pipeline config file
# onemod --config /path/to/pipeline/config.json --stage_name stage_name --from_pipeline --method predict

# Predict stage from command line using stage config file
# onemod --config /path/to/stage/config.json --method predict
