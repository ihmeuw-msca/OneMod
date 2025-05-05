import json

import pytest
from tests.helpers.dummy_pipeline import setup_dummy_pipeline
from tests.helpers.dummy_stages import assert_stage_logs
from tests.helpers.get_expected_args import get_expected_args
from tests.helpers.utils import assert_equal_unordered

KWARGS = {
    "backend": "local",
    "cluster": None,
    "resources": None,
    "python": None,
    "subsets": None,
    "paramsets": None,
    "collect": None,
}


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_dummy_pipeline(small_input_data, test_base_dir, method):
    """End-to-end test for a the OneMod example pipeline with arbitrary configs and constraints, test data."""
    # Setup the pipeline
    dummy_pipeline, stages = setup_dummy_pipeline(
        small_input_data, test_base_dir
    )

    # Validate, build, and save the pipeline
    pipeline_json_path = test_base_dir / f"{dummy_pipeline.name}.json"
    dummy_pipeline.build()  # Saves to pipeline_json_path by default

    # Read in built pipeline representation
    with open(pipeline_json_path, "r") as f:
        dummy_pipeline_dict = json.load(f)

    assert dummy_pipeline_dict["name"] == "dummy_pipeline"
    assert dummy_pipeline_dict["directory"] == str(test_base_dir)
    assert dummy_pipeline_dict["groupby_data"] == str(small_input_data)
    assert_equal_unordered(
        dummy_pipeline_dict["config"],
        {
            "id_columns": ["age_group_id", "location_id", "year_id", "sex_id"],
            "model_type": "binomial",
        },
    )
    assert_equal_unordered(
        dummy_pipeline_dict["dependencies"],
        {
            "preprocessing": [],
            "covariate_selection": ["preprocessing"],
            "global_model": ["covariate_selection", "preprocessing"],
            "location_model": ["preprocessing", "global_model"],
            "smoothing": ["preprocessing", "location_model"],
            "custom_stage": ["smoothing", "preprocessing"],
        },
    )

    # Run the pipeline with the given method (run, fit, predict)
    dummy_pipeline.evaluate(
        method=method,
        stages=None,
        backend="local",
        cluster=None,
        resources=None,
        python=None,
    )

    # Check each stage's log output for correct method calls on correct subsets/paramsets
    expected_args = get_expected_args()

    for stage in stages:
        if stage.name in expected_args:
            assert_stage_logs(
                stage,
                expected_args[stage.name]["methods"][method],
                expected_args[stage.name]["subsets"],
                expected_args[stage.name]["paramsets"],
            )
        else:
            assert False, "Unknown stage name"
