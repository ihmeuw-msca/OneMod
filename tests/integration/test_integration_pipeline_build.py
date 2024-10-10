from pathlib import Path

import pytest

from onemod.config import Config, PipelineConfig, StageConfig
from onemod.pipeline import Pipeline
from onemod.stage import Stage
from onemod.types import Data


class DummyStage(Stage):
    config: Config
    _required_input: set[str] = {"input_data.parquet"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}

@pytest.fixture(scope="module")
def sample_stage_dir(tmp_path_factory):
    sample_stage_dir = tmp_path_factory.mktemp("example_stage")
    return sample_stage_dir

@pytest.fixture(scope="module")
def example_stage(sample_stage_dir):
    example_stage = DummyStage(
        name="example_stage",
        directory=sample_stage_dir,
        config=StageConfig(ids={"age_group_id", "location_id"}),
        input_types=dict(
            input_data=Data(
                stage="example_stage",
                path=Path(sample_stage_dir, "input_data.parquet"),
                format="parquet",
                shape=(1, 2),
                columns=dict(
                    age_group_id=dict(type=int),
                    location_id=dict(type=int),
                )
            )
        ),
        output_types=dict(
            predictions=Data(
                stage="example_stage",
                path=Path(sample_stage_dir, "predictions.parquet"),
                format="parquet",
                shape=(3, 4),
                columns=dict(
                    age_group_id=dict(type=int),
                    location_id=dict(type=int),
                )
            )
        )
    )
    
    return example_stage

@pytest.fixture(scope="module")
def pipeline_with_single_stage(sample_stage_dir, example_stage):
    """A sample pipeline with a single stage and no dependencies."""
    pipeline = Pipeline(
        name="test_pipeline",
        config=PipelineConfig(ids=["age_group_id", "location_id"]),
        directory=sample_stage_dir,
        groupby=["age_group_id"]
    )
    print(example_stage)
    pipeline.add_stage(example_stage)
    example_stage(input_data=sample_stage_dir / "input_data.parquet")
    
    return pipeline

@pytest.mark.integration
def test_pipeline_build_single_stage(sample_stage_dir, pipeline_with_single_stage):
    """Test building a pipeline with a single stage and no dependencies."""
    pipeline_dict = pipeline_with_single_stage.build()

    # Check high-level pipeline metadata
    assert pipeline_dict["name"] == "test_pipeline"
    assert Path(pipeline_dict["directory"]) == sample_stage_dir
    assert pipeline_dict["groupby"] == {"age_group_id"}

    # Check that the stage is serialized correctly
    assert "example_stage" in pipeline_dict["stages"]
    assert pipeline_dict["stages"]["example_stage"]["name"] == "example_stage"
    assert pipeline_dict["stages"]["example_stage"]["config"]["ids"] == {"age_group_id", "location_id"}

    # Check that dependencies are empty
    assert pipeline_dict["dependencies"] == {"example_stage": set()}
