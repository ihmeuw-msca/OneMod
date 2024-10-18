import io
from contextlib import redirect_stdout
from pathlib import Path
import json
from typing import List

import pytest

from onemod import Pipeline
from onemod.constraints import Constraint
from onemod.dtypes import ColumnSpec, Data
from onemod.stage import Stage
from tests.helpers.dummy_stages import DummyCustomStage, DummyKregStage, DummyPreprocessingStage, DummyRoverStage, DummySpxmodStage

from tests.helpers.utils import assert_equal_unordered


@pytest.fixture(scope="module")
def test_base_dir(tmp_path_factory):
    test_base_dir = tmp_path_factory.mktemp("example")
    return test_base_dir


@pytest.fixture(scope="module")
def test_input_data(test_assets_dir):
    test_input_data_path = Path(
        test_assets_dir, "e2e", "example1", "data", "small_data.parquet"
    )
    return test_input_data_path


def setup_dummy_pipeline(test_input_data, test_base_dir):
    """Set up a dummy pipeline, including specific dummy stages."""
    preprocessing = DummyPreprocessingStage(
        name="preprocessing",
        config={},
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
                            Constraint(
                                name="bounds",
                                args={"ge": 0, "le": 1}
                            )
                        ]
                    ),
                }
            )
        }
    )
    covariate_selection = DummyRoverStage(
        name="covariate_selection",
        config={"cov_exploring": ["cov1", "cov2", "cov3"]},
        groupby=["age_group_id"],
    )
    global_model = DummySpxmodStage(
        name="global_model",
        config={"xmodel": {"variables": [{"name": "var1"}, {"name": "var2"}]}},
    )
    location_model = DummySpxmodStage(
        name="location_model",
        config={"xmodel": {"variables": [{"name": "var1"}, {"name": "var2"}]}},
        groupby=["location_id"],
    )
    smoothing = DummyKregStage(
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
    custom_stage = DummyCustomStage(
        name="custom_stage",
        config={"custom_param": [1, 2]},
        groupby=["super_region_id"],
    )

    # Create pipeline
    dummy_pipeline = Pipeline(
        name="dummy_pipeline",
        config={
            "id_columns": ["age_group_id", "location_id", "sex_id", "year_id"],
            "model_type": "binomial",
        },
        directory=test_base_dir,
        data=test_input_data,
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
            custom_stage,
        ]
    )
    
    # Define dependencies
    preprocessing(data=dummy_pipeline.data)
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
    
    return dummy_pipeline, [preprocessing, covariate_selection, global_model, location_model, smoothing, custom_stage]
    

def assert_stage_logs(
    stage: Stage,
    methods: List[str] | None = None,
    subset_ids: List[int] | None = None,
    param_ids: List[int] | None =None
):
    """Assert that the expected methods were logged for a given stage."""
    log = stage.get_log()
    if methods:
        for method in methods:
            for subset_id in subset_ids:
                if param_ids:
                    for param_id in param_ids:
                        assert f"{method}: name={stage.name}, subset={subset_id}, param={param_id}" in log
                else:
                    assert f"{method}: name={stage.name}, subset={subset_id}, param=None" in log
        assert f"collect: name={stage.name}" in log


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_dummy_pipeline(test_input_data, test_base_dir, method):
    """End-to-end test for a the OneMod example pipeline with arbitrary configs and constraints, test data."""
    # Setup the pipeline
    dummy_pipeline, stages = setup_dummy_pipeline(test_input_data, test_base_dir)
    
    # Validate, build, and save the pipeline
    pipeline_json_path = test_base_dir / f"{dummy_pipeline.name}.json"
    dummy_pipeline.build()  # Saves to pipeline_json_path by default
    
    # Read in built pipeline representation
    with open(pipeline_json_path, "r") as f:
        dummy_pipeline_dict = json.load(f)
    
    assert dummy_pipeline_dict["name"] == "dummy_pipeline"
    assert dummy_pipeline_dict["directory"] == str(test_base_dir)
    assert dummy_pipeline_dict["data"] == str(test_input_data)
    assert dummy_pipeline_dict["groupby"] == ["sex_id"]
    assert_equal_unordered(
        dummy_pipeline_dict["config"],
        {
            "id_columns": ["age_group_id", "location_id", "year_id", "sex_id"],
            "model_type": "binomial",
            "observation_column": "obs",
            "prediction_column": "pred",
            "weight_column": "weights",
            "test_column": "test",
            "holdout_columns": [],
            "coef_bounds": {}
        }
    )
    assert_equal_unordered(
        dummy_pipeline_dict["dependencies"],
        {
            "preprocessing": [],
            "covariate_selection": ["preprocessing"],
            "global_model": ["covariate_selection", "preprocessing"],
            "location_model": ["preprocessing", "global_model"],
            "smoothing": ["preprocessing", "location_model"],
            "custom_stage": ["smoothing", "preprocessing"]
        }
    )
    
    # Run the pipeline with the given method (run, fit, predict)
    dummy_pipeline.evaluate(backend="local", method=method)

    # Set expected methods, subset ids, param ids for each stage
    expected_args = {
        "covariate_selection": {
            "methods": {
                "run": ["run", "fit"],
                "fit": ["fit"],
                "predict": None,
            },
            "subset_ids": range(3),
            "param_ids": None,
        },
        "global_model": {
            "methods": {
                "run": ["run", "fit", "predict"],
                "fit": ["fit"],
                "predict": ["predict"],
            },
            "subset_ids": range(2),
            "param_ids": None,
        },
        "location_model": {
            "methods": {
                "run": ["run", "fit", "predict"],
                "fit": ["fit"],
                "predict": ["predict"],
            },
            "subset_ids": range(4),
            "param_ids": None,
        },
        "smoothing": {
            "methods": {
                "run": ["run", "fit", "predict"],
                "fit": ["fit"],
                "predict": ["predict"],
            },
            "subset_ids": range(4),
            "param_ids": None,
        },
        "custom_stage": {
            "methods": {
                "run": ["run", "fit", "predict"],
                "fit": ["fit"],
                "predict": ["predict"],
            },
            "subset_ids": range(4),
            "param_ids": range(2),
        },
    }
        
    # Check each stage's log output for correct method calls on correct subset/param ids
    for stage in stages:
        if stage.name == "preprocessing":
            if method in ["run", "fit"]:
                assert stage.get_log() == [f"run: name={stage.name}"]
            else:
                assert stage.get_log() == []
        elif stage.name in expected_args:
            assert_stage_logs(stage, expected_args[stage.name]["methods"][method], expected_args[stage.name]["subset_ids"], expected_args[stage.name]["param_ids"])
        else:
            assert False, "Unknown stage name"
