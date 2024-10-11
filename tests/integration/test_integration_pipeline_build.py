from pathlib import Path

import pytest

from onemod.config import Config, PipelineConfig, StageConfig
from onemod.constraints import Constraint
from onemod.pipeline import Pipeline
from onemod.stage import Stage
from onemod.types import Data


class DummyStage(Stage):
    config: Config
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}

@pytest.fixture(scope="module")
def test_base_dir(tmp_path_factory):
    test_base_dir = tmp_path_factory.mktemp("test_base_dir")
    return test_base_dir

@pytest.fixture(scope="module")
def stage_1(test_base_dir):
    stage_1 = DummyStage(
        name="stage_1",
        directory=test_base_dir / "stage_1",
        config=StageConfig(),
        input_types=dict(
            data=Data(
                stage="stage_1",
                path=test_base_dir / "stage_1" / "data.parquet",
                format="parquet",
                columns=dict(
                    id_col=dict(type=int),
                    bounded_col=dict(
                        type=float,
                        constraints=[
                            Constraint("bounds", ge=0, le=1)
                        ]
                    ),
                    str_col=dict(
                        type=str,
                        constraints=[
                            Constraint("is_in", other={"a", "b", "c"})
                        ]
                    )
                )
            ),
            covariates=Data(
                stage="stage_1",
                path=test_base_dir / "stage_1" / "covariates.csv",
                format="csv",
                columns=dict(
                    id_col=dict(type=int),
                    str_col=dict(
                        type=str,
                        constraints=[
                            Constraint("is_in", other=["cov1", "cov2", "cov3"])
                        ]
                    )
                )
            )
        ),
        output_types=dict(
            predictions=Data(
                stage="stage_1",
                path=test_base_dir / "stage_1" / "predictions.parquet",
                format="parquet",
                columns=dict(
                    id_col=dict(type=int),
                    prediction_col=dict(
                        type=float,
                        constraints=[
                            Constraint("bounds", ge=-1, le=1)
                        ]
                    )
                )
            )
        )
    )
    stage_1.directory = test_base_dir / "stage_1"
    stage_1(
        data=test_base_dir / "stage_1" / "data.parquet",
        covariates=test_base_dir / "stage_1" / "covariates.csv"
    )
    print(stage_1.to_dict())
    return stage_1

@pytest.fixture(scope="module")
def pipeline_with_single_stage(test_base_dir, stage_1):
    """A sample pipeline with a single stage and no dependencies."""
    pipeline = Pipeline(
        name="test_pipeline",
        config=PipelineConfig(ids=["age_group_id", "location_id"]),
        directory=test_base_dir,
        groupby=["age_group_id"]
    )
    pipeline.add_stage(stage_1)
    
    return pipeline

@pytest.mark.integration
def test_pipeline_build_single_stage(test_base_dir, pipeline_with_single_stage):
    """Test building a pipeline with a single stage and no dependencies."""
    pipeline_dict = pipeline_with_single_stage.build()
    print(pipeline_dict)
    
    assert pipeline_dict == {
        "name": "test_pipeline",
        "directory": str(test_base_dir),
        "ids": {"age_group_id", "location_id"},
        "groupby": {"age_group_id"},
        "obs": "obs",
        "pred": "pred",
        "weights": "weights",
        "test": "test",
        "holdouts": set(),
        "mtype": "binomial",
        "stages": {
            "example_stage": {
                "name": "example_stage",
                "type": "DummyStage",
                "config": {
                    "ids": {"age_group_id", "location_id"},
                    "obs": "obs",
                    "pred": "pred",
                    "weights": "weights",
                    "test": "test",
                    "holdouts": set(),
                    "mtype": "binomial"
                },
                "input": {
                    "data": {
                        "stage": "example_stage",
                        "path": str(test_base_dir / "data.parquet"),
                        "format": "parquet",
                        "shape": (1, 2),
                        "columns": {
                            "age_group_id": {"type": "int"},
                            "location_id": {"type": "int"},
                        }
                    }
                },
                "output": {
                    "predictions": {
                        "stage": "example_stage",
                        "path": str(test_base_dir / "predictions.parquet"),
                        "format": "parquet",
                        "shape": (3, 4),
                        "columns": {
                            "age_group_id": {"type": "int"},
                            "location_id": {"type": "int"},
                        }
                    },
                    "model": {
                        "stage": "example_stage",
                        "path": str(test_base_dir / "model.pkl"),
                        "format": "parquet",
                        "shape": None,
                        "columns": None
                    }
                },
                "dependencies": set()
            }
        },
        "dependencies": {"example_stage": set()}
    }
