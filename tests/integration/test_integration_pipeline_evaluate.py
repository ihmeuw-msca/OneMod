from unittest.mock import patch

import polars as pl
import pytest
from tests.helpers.dummy_pipeline import get_expected_args, setup_dummy_pipeline
from tests.helpers.dummy_stages import assert_stage_logs
from tests.helpers.utils import assert_equal_unordered

from onemod.config import PipelineConfig, StageConfig
from onemod.pipeline import Pipeline
from onemod.stage import Stage


class MultiplyByTwoStage(Stage):
    """Stage that multiplies the value column by 2."""

    config: StageConfig = {}
    _skip: set[str] = {"predict"}
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {
        "age_metadata.parquet",
        "location_metadata.parquet",
    }
    _output: set[str] = {"data.parquet"}

    def run(self) -> None:
        """Run MultiplyByTwoStage."""
        lf = self.dataif.load(key="data", lazy=True)
        df = lf.with_columns((pl.col("value") * 2).alias("value")).collect()
        self.dataif.dump(df, "data.parquet", key="output")


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

    subset_stage_names = {"preprocessing", "covariate_selection"}
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


@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_missing_dependency_error(small_input_data, test_base_dir, method):
    """Test that Pipeline.evaluate() on a subset of stages raises an error when required inputs for a specified stage are missing."""
    dummy_pipeline, stages = setup_dummy_pipeline(
        small_input_data, test_base_dir
    )

    subset_stage_names = {"covariate_selection"}

    with pytest.raises(
        ValueError,
        match="Required input to stage 'covariate_selection' is missing. Missing output from upstream dependency 'preprocessing'.",
    ):
        dummy_pipeline.evaluate(method=method, stages=subset_stage_names)


@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_invalid_id_subsets_keys(small_input_data, test_base_dir, method):
    """Test that Pipeline.evaluate() raises an error when an invalid id_subsets key is provided."""
    dummy_pipeline, stages = setup_dummy_pipeline(
        small_input_data, test_base_dir
    )

    subset_stage_names = {"preprocessing"}

    with pytest.raises(
        ValueError,
        match="id_subsets keys {'invalid_id_col_name'} do not match groupby columns {'sex_id'}",
    ):
        dummy_pipeline.evaluate(
            method=method,
            stages=subset_stage_names,
            id_subsets={"invalid_id_col_name": [1, 2, 3]},
        )


@pytest.mark.integration
# @pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_evaluate_with_id_subsets(test_base_dir, sample_data):
    """Test that Pipeline.evaluate() correctly evaluates single stage with id_subsets."""
    sample_input_data = test_base_dir / "test_input_data.parquet"
    df = pl.DataFrame(sample_data)
    df.write_parquet(sample_input_data)

    test_pipeline = Pipeline(
        name="dummy_pipeline",
        config=PipelineConfig(
            id_columns=["age_group_id", "location_id", "sex_id"],
            model_type="binomial",
        ),
        directory=test_base_dir,
        data=sample_input_data,
        groupby={"age_group_id"},
    )
    test_stage = MultiplyByTwoStage(
        name="multiply_by_two", config=StageConfig()
    )
    test_pipeline.add_stages([test_stage])
    test_stage(data=test_pipeline.data)

    # Ensure input data is as expected for the test
    assert sample_input_data.exists()
    input_df = pl.read_parquet(sample_input_data)
    print(input_df.columns)

    test_pipeline.evaluate(method="run", id_subsets={"age_group_id": [1]})

    # Verify that output only contains rows with age_group_id == 1
    print(test_stage.dataif.get_path(key="data"))
    print(test_stage.dataif.get_path(key="output"))
    print(test_stage.dataif.get_path(key="output").exists())
    output_df = pl.read_parquet(test_stage.dataif.get_path(key="output"))
    # output_df = test_stage.dataif.load(key="output")  # Is this how we want to access, or do we want to require specification of fparts, i.e. there can be more than one path per key? But seems like 1:1 makes sense if we are doing paths instead of directories. Friday night work comment. May become suddenly clearer in the morning.
    print(output_df.shape)
    print(output_df.columns)
    print(output_df.select(pl.col("age_group_id")).n_unique())
    assert output_df.select(pl.col("age_group_id")).n_unique() == 1


@pytest.mark.integration
@pytest.mark.requires_data
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
