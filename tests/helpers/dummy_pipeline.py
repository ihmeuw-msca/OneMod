from tests.helpers.dummy_stages import (
    DummyCustomConfig,
    DummyCustomStage,
    DummyKregConfig,
    DummyKregStage,
    DummyPreprocessingStage,
    DummyRoverConfig,
    DummyRoverStage,
    DummySpxmodConfig,
    DummySpxmodStage,
)

from onemod import Pipeline
from onemod.config import Config, StageConfig
from onemod.constraints import Constraint
from onemod.dtypes import ColumnSpec, Data


def setup_dummy_pipeline(test_input_data, test_base_dir):
    """Set up a dummy pipeline, including specific dummy stages."""
    preprocessing = DummyPreprocessingStage(
        name="preprocessing",
        config=StageConfig(),
        input_validation={
            "data": Data(
                stage="input_data",
                path=test_input_data,
                format="parquet",
                shape=(5, 52),
                columns={
                    "adult_hiv_death_rate": ColumnSpec(
                        type="float",
                        constraints=[
                            Constraint(name="bounds", args={"ge": 0, "le": 1})
                        ],
                    )
                },
            )
        },
    )
    covariate_selection = DummyRoverStage(
        name="covariate_selection",
        config=DummyRoverConfig(
            model_type="binomial",
            observation_column="fake_observation_column",
            weights_column="fake_weights_column",
            holdout_columns=["holdout1", "holdout2", "holdout3"],
            cov_exploring=["cov1", "cov2", "cov3"],
            cov_groupby=["sex_id"],
        ),
        groupby=["sex_id", "age_group_id"],
    )
    global_model = DummySpxmodStage(
        name="global_model",
        config=DummySpxmodConfig(
            id_columns=["age_group_id", "location_id", "sex_id", "year_id"],
            model_type="binomial",
            observation_column="fake_observation_column",
            prediction_column="fake_prediction_column",
            weights_column="fake_weights_column",
            xmodel={"variables": [{"name": "var1"}, {"name": "var2"}]},
        ),
        groupby=["sex_id"],
    )
    location_model = DummySpxmodStage(
        name="location_model",
        config=DummySpxmodConfig(
            id_columns=["age_group_id", "location_id", "sex_id", "year_id"],
            model_type="binomial",
            observation_column="fake_observation_column",
            prediction_column="fake_prediction_column",
            weights_column="fake_weights_column",
            xmodel={"variables": [{"name": "var1"}, {"name": "var2"}]},
        ),
        groupby=["sex_id", "location_id"],
    )
    smoothing = DummyKregStage(
        name="smoothing",
        config=DummyKregConfig(
            id_columns=["age_group_id", "location_id", "sex_id", "year_id"],
            model_type="binomial",
            kreg_model={
                "age_scale": 1,
                "gamma_age": 1,
                "gamma_year": 1,
                "exp_location": 1,
                "lam": 1,
                "nugget": 1,
            },
            kreg_fit={
                "gtol": 1,
                "max_iter": 1,
                "cg_maxiter": 1,
                "cg_maxiter_increment": 1,
                "nystroem_rank": 1,
            },
        ),
        groupby=["sex_id", "region_id"],
    )
    custom_stage = DummyCustomStage(
        name="custom_stage",
        config=DummyCustomConfig(custom_param=[1, 2]),
        groupby=["sex_id", "super_region_id"],
        crossby=["custom_param"],
    )

    # Create pipeline
    dummy_pipeline = Pipeline(
        name="dummy_pipeline",
        config=Config(
            id_columns=["age_group_id", "location_id", "sex_id", "year_id"],
            model_type="binomial",
        ),
        directory=test_base_dir,
        groupby_data=test_input_data,
    )

    # Add stages
    dummy_pipeline.add_stages(
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
    preprocessing(data=dummy_pipeline.groupby_data)
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

    dummy_pipeline.build()

    return dummy_pipeline, [
        preprocessing,
        covariate_selection,
        global_model,
        location_model,
        smoothing,
        custom_stage,
    ]
