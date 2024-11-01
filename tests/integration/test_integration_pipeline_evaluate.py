import pytest
from tests.helpers.dummy_pipeline import get_expected_args, setup_dummy_pipeline
from tests.helpers.dummy_stages import assert_stage_logs


@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_subset_stage_identification(small_input_data, test_base_dir, method):
    """Test that Pipeline.evaluate() identifies the correct subset of stages."""
    dummy_pipeline, stages = setup_dummy_pipeline(
        small_input_data, test_base_dir
    )

    subset_stage_names = {"preprocessing", "covariate_selection"}
    subset_stages = [
        stage for stage in stages if stage.name in subset_stage_names
    ]

    try:
        dummy_pipeline.evaluate(method=method, stages=subset_stage_names)
    except Exception as e:
        pytest.fail(f"evaluate() raised an unexpected exception: {e}")

    # Check that the subset of stages was evaluated
    expected_args = get_expected_args()

    for stage in subset_stages:
        if stage.name == "preprocessing":
            if method in ["run", "fit"]:
                assert stage.get_log() == [f"run: name={stage.name}"]
            else:
                assert stage.get_log() == []
        elif stage.name in expected_args:
            assert_stage_logs(
                stage,
                expected_args[stage.name]["methods"][method],
                expected_args[stage.name]["subset_ids"],
                expected_args[stage.name]["param_ids"],
            )
        else:
            assert False, "Unknown stage name"

    # Check that other stages were not evaluated
    for stage in stages:
        if stage.name not in subset_stage_names:
            assert stage.get_log() == []

    # ## TODO: Tests for evaluating subsets of stages
    # 1. Done
    # # 2. Covariate selection only, only valid if preprocessing output exists
    # # Ensure preprocessing.output["data"] path does not exist, delete if it does
    # if Path(preprocessing.output["data"]).exists():  # noqa
    #     Path(preprocessing.output["data"]).unlink()  # noqa
    # # with pytest.raises... tbd error

    # # 3. Preprocessing, then covariate selection (valid)
    # dummy_pipeline.evaluate(backend="local", method=method, stages=["preprocessing"])
    # dummy_pipeline.evaluate(backend="local", method=method, stages=["covariate_selection"])

    # # 4. Preprocessing and covariate selection (valid)
    # dummy_pipeline.evaluate(backend="local", method=method, stages=["preprocessing", "covariate_selection"])
