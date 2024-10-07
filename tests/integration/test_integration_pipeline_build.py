from pathlib import Path

import pytest

from onemod.config import PipelineConfig
from onemod.io import Input, Output
from onemod.pipeline import Pipeline
from onemod.stage import Stage
from onemod.types import Data

@pytest.fixture
def sample_stage(tmp_path):
    """A sample fixture for a dummy stage."""
    return Stage(
        name="example_stage",
        config={"param1": "value1"},
        input=Input(
            stage="example_stage",
            required=dict(
                input_data=Data(
                    stage="example_stage",
                    path=Path(tmp_path / "data.parquet"),
                    format="parquet",
                    shape=(1, 2),
                    columns=dict(
                        age_group_id=dict(type=int),
                        location_id=dict(type=int),
                    )
                )
            )
        ),
        output=Output(
            stage="example_stage",
            data=dict(
                output_data=Data(
                    stage="example_stage",
                    path=Path(tmp_path / "output.parquet"),
                    format="parquet",
                    shape=(3, 4),
                    columns=dict(
                        age_group_id=dict(type=int),
                        location_id=dict(type=int),
                    )
                )
            )
        )
    )

@pytest.fixture
def pipeline_with_single_stage(tmp_path, sample_stage):
    """A sample pipeline with a single stage and no dependencies."""
    pipeline = Pipeline(
        name="test_pipeline",
        config=PipelineConfig(ids=["age_group_id", "location_id"]),
        directory=tmp_path,
        groupby=["age_group_id"]
    )
    pipeline.add_stage(sample_stage, dependencies=[])
    return pipeline

@pytest.mark.integration
def test_pipeline_build_single_stage(tmp_path, pipeline_with_single_stage):
    """Test building a pipeline with a single stage and no dependencies."""
    pipeline_dict = pipeline_with_single_stage.build()

    # Check high-level pipeline metadata
    assert pipeline_dict["name"] == "test_pipeline"
    assert Path(pipeline_dict["directory"]) == tmp_path
    assert pipeline_dict["groupby"] == {"age_group_id"}

    # Check that the stage is serialized correctly
    assert "example_stage" in pipeline_dict["stages"]
    assert pipeline_dict["stages"]["example_stage"]["name"] == "example_stage"
    assert pipeline_dict["stages"]["example_stage"]["config"]["param1"] == "value1"

    # Check that dependencies are empty
    assert pipeline_dict["dependencies"] == {"example_stage": []}
