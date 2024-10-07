from pathlib import Path

import pytest

from onemod.pipeline import Pipeline
from onemod.stage import Stage
from onemod.types import Data

@pytest.fixture
def sample_stage(tmp_path):
    """A sample fixture for a dummy stage."""
    return Stage(
        name="example_stage",
        config={"param1": "value1"},
        input=dict(
            required=dict(
                input_data=dict(
                    type=Data,
                    data=dict(
                        stage="example_stage",
                        path=Path(tmp_path / "data.parquet"),
                        format="parquet",
                        shape=(100, 10),
                        columns={
                            "col1": {"type": int},
                            "col2": {"type": float},
                            "col3": {"type": str},
                        }
                    )
                )
            )
        ),
        output=[Data]
    )

@pytest.fixture
def pipeline_with_single_stage(tmp_path, sample_stage):
    """A sample pipeline with a single stage and no dependencies."""
    pipeline = Pipeline(
        name="test_pipeline",
        directory=tmp_path,
        groupby=["age_group_id"]
    )
    pipeline._stages = {"example_stage": sample_stage}
    return pipeline

@pytest.mark.integration
def test_pipeline_build_single_stage(tmp_path, pipeline_with_single_stage):
    """Test building a pipeline with a single stage and no dependencies."""
    pipeline_dict = pipeline_with_single_stage.build()

    # Check high-level pipeline metadata
    assert pipeline_dict["name"] == "test_pipeline"
    assert pipeline_dict["directory"] == tmp_path
    assert pipeline_dict["groupby"] == ["age_group_id"]

    # Check that the stage is serialized correctly
    assert "example_stage" in pipeline_dict["stages"]
    assert pipeline_dict["stages"]["example_stage"]["name"] == "example_stage"
    assert pipeline_dict["stages"]["example_stage"]["config"]["param1"] == "value1"

    # Check that dependencies are empty
    assert pipeline_dict["dependencies"] == {}

    # Check default execution metadata (placeholders)
    assert pipeline_dict["execution"]["tool"] == "sequential"
    assert pipeline_dict["execution"]["cluster_name"] is None
