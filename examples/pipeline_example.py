"""Example OneMod pipeline."""

import fire
from custom_stage import CustomStage

from onemod import Pipeline
from onemod.stage import KregStage, RoverStage, SpxmodStage


def create_pipeline(directory: str, data: str):
    # Create stages
    # Stage-specific validation specifications go here.
    # Stage classes may also implement default validation specifications.
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
        groupby_data=data,
        groupby=["sex_id"],
    )

    # Add stages
    example_pipeline.add_stages(
        [
            covariate_selection,
            global_model,
            location_model,
            smoothing,
            custom_stage,
        ]
    )

    # Define dependencies
    covariate_selection(data=data)
    global_model(
        data=data, selected_covs=covariate_selection.output["selected_covs"]
    )
    location_model(data=data, offset=global_model.output["predictions"])
    smoothing(data=data, offset=location_model.output["predictions"])
    custom_stage(observations=data, predictions=smoothing.output["predictions"])

    # Serialize pipeline
    example_pipeline.to_json()

    # User could call this method themself, but evaluate() also
    # calls it in case updates have been made to the
    # pipeline (e.g., someone is experimenting with a pipeline in a
    # a notebook)
    example_pipeline.build()

    # Run (fit and predict) entire pipeline
    example_pipeline.evaluate(method="run")

    # Fit specific stages
    example_pipeline.evaluate(
        method="fit", stages=["covariate_selection", "global_model"]
    )

    # Predict for specific locations
    example_pipeline.evaluate(
        method="predict", id_subsets={"location_id": [1, 2, 3]}
    )


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
