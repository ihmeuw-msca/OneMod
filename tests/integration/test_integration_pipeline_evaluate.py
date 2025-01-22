from unittest.mock import patch

import pytest
from tests.helpers.dummy_pipeline import get_expected_args, setup_dummy_pipeline
from tests.helpers.dummy_stages import assert_stage_logs
from tests.helpers.utils import assert_equal_unordered

# FIXME: WRONG DATA SET


@pytest.fixture
def sample_data():
    return {
        "age_group_id": [1, 2, 2, 3],
        "location_id": [10, 20, 20, 30],
        "sex_id": [1, 2, 1, 2],
        "value": [100, 200, 300, 400],
    }


def create_dummy_preprocessing_output_file(test_base_dir, stages):
    """Create a dummy preprocessing output file."""
    preprocessing = [
        stage for stage in stages if stage.name == "preprocessing"
    ][0]
    preprocessing_output = (
        test_base_dir / preprocessing.name / preprocessing.output["data"].path
    )
    preprocessing_output.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_output.touch()


@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_invalid_stage_name(small_input_data, test_base_dir, method):
    """Test that Pipeline.evaluate() raises an error when an invalid stage name is provided."""
    dummy_pipeline, stages = setup_dummy_pipeline(
        small_input_data, test_base_dir
    )

    with pytest.raises(
        ValueError, match="Stage 'invalid_stage_name' not found"
    ):
        dummy_pipeline.evaluate(method=method, stages=["invalid_stage_name"])

    with pytest.raises(
        ValueError, match="Stage 'invalid_stage_name' not found"
    ):
        dummy_pipeline.evaluate(
            method=method,
            stages=[
                "preprocessing",
                "invalid_stage_name",
                "covariate_selection",
            ],
        )


@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_subset_stage_identification(small_input_data, test_base_dir, method):
    """Test that Pipeline.evaluate() identifies the correct subset of stages."""
    dummy_pipeline, stages = setup_dummy_pipeline(
        small_input_data, test_base_dir
    )

    subset_stage_names = ["preprocessing", "covariate_selection"]
    subset_stages = [
        stage for stage in stages if stage.name in subset_stage_names
    ]

    # Manually write dummy preprocessing output data file
    create_dummy_preprocessing_output_file(test_base_dir, stages)

    try:
        dummy_pipeline.evaluate(method=method, stages=subset_stage_names)
    except Exception as e:
        pytest.fail(f"evaluate() raised an unexpected exception: {e}")

    # Check that the subset of stages was evaluated
    expected_args = get_expected_args()

    for stage in subset_stages:
        if stage.name in expected_args:
            assert_stage_logs(
                stage,
                expected_args[stage.name]["methods"][method],
                expected_args[stage.name]["subsets"],
                expected_args[stage.name]["paramsets"],
            )
        else:
            assert False, "Unknown stage name"

    # Check that other stages were not evaluated
    for stage in stages:
        if stage.name not in subset_stage_names:
            assert stage.get_log() == []


@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_missing_dependency_error(small_input_data, test_base_dir, method):
    """Test that Pipeline.evaluate() on a subset of stages raises an error when required inputs for a specified stage are missing."""
    dummy_pipeline, stages = setup_dummy_pipeline(
        small_input_data, test_base_dir
    )

    subset_stage_names = ["covariate_selection"]

    with pytest.raises(
        FileNotFoundError,
        match=f"Stage 'covariate_selection' input items do not exist: {{'data': '{test_base_dir}/preprocessing/data.parquet'}}",
    ):
        dummy_pipeline.evaluate(method=method, stages=subset_stage_names)


@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_duplicate_stage_names(small_input_data, test_base_dir, method):
    """Test that duplicate stage names passed to Pipeline.evaluate() are coerced to unique stage names."""
    dummy_pipeline, stages = setup_dummy_pipeline(
        small_input_data, test_base_dir
    )

    subset_stage_names = ["preprocessing", "preprocessing"]

    dummy_pipeline.evaluate(method=method, stages=subset_stage_names)

    # Check that preprocessing was evaluated only once
    assert (
        len([stage for stage in stages if stage.name == "preprocessing"]) == 1
    )
    for stage in stages:
        if stage.name == "preprocessing":
            if method in ["run", "fit"]:
                assert stage.get_log() == [
                    f"run: name={stage.name}, subset=None, paramset=None"
                ]
            else:
                assert stage.get_log() == []
        else:
            assert stage.get_log() == []


@pytest.mark.skip(
    "Jobmon test - Run manually until better jobmon testing solution in place"
)
@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_evaluate_with_jobmon_subset_call(
    dummy_resources, small_input_data, test_base_dir, method
):
    """Test that evaluate_with_jobmon is called with the correct subset."""
    assert dummy_resources.exists()

    dummy_pipeline, stages = setup_dummy_pipeline(
        small_input_data, test_base_dir
    )

    subset_stage_names = {"preprocessing", "covariate_selection"}

    with patch(
        "onemod.backend.evaluate_with_jobmon"
    ) as mock_evaluate_with_jobmon:
        dummy_pipeline.evaluate(
            backend="jobmon",
            cluster="local",
            resources=dummy_resources,
            method=method,
            stages=subset_stage_names,
        )

        mock_evaluate_with_jobmon.assert_called_once()

        args, kwargs = mock_evaluate_with_jobmon.call_args

        assert kwargs["model"] == dummy_pipeline
        assert kwargs["cluster"] == "local"
        assert kwargs["resources"] == dummy_resources
        assert kwargs["method"] == method
        assert_equal_unordered(kwargs["stages"], list(subset_stage_names))
