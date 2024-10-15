import json
from pathlib import Path

from polars import DataFrame
import pytest

from onemod.config import PipelineConfig, StageConfig
from onemod.constraints import Constraint
from onemod.pipeline import Pipeline
from onemod.stage import Stage
from onemod.types import Data, ColumnSpec


class DummyStage(Stage):
    config: StageConfig
    _required_input: set[str] = {"data.parquet", "covariates.parquet"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}

def assert_equal_unordered(actual, expected):
    """Recursively compare two data structures, treating lists as unordered collections."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert set(actual.keys()) == set(expected.keys()), f"Dict keys differ: {actual.keys()} != {expected.keys()}"
        for key in actual:
            assert_equal_unordered(actual[key], expected[key])
    elif isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), f"List lengths differ: {len(actual)} != {len(expected)}"
        unmatched_expected_items = expected.copy()
        for actual_item in actual:
            match_found = False
            for expected_item in unmatched_expected_items:
                try:
                    assert_equal_unordered(actual_item, expected_item)
                    unmatched_expected_items.remove(expected_item)
                    match_found = True
                    break
                except AssertionError:
                    continue
            if not match_found:
                raise AssertionError(f"No matching item found for {actual_item} in expected list.")
        if unmatched_expected_items:
            raise AssertionError(f"Expected items not matched: {unmatched_expected_items}")
    else:
        assert actual == expected, f"Values differ: {actual} != {expected}"

@pytest.fixture(scope="module")
def test_base_dir(tmp_path_factory):
    test_base_dir = tmp_path_factory.mktemp("test_base_dir")
    return test_base_dir

@pytest.fixture(scope="module")
def create_dummy_data(test_base_dir):
    """Create dummy data files needed for testing."""
    data_dir = test_base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_df = DataFrame({
        "id_col": [1],
        "bounded_col": [0.5],
        "str_col": ["a"]
    })
    data_parquet_path = data_dir / "data.parquet"
    data_df.write_parquet(data_parquet_path)

    covariates_df = DataFrame({
        "id_col": [1],
        "str_col": ["cov1"]
    })
    covariates_parquet_path = data_dir / "covariates.parquet"
    covariates_df.write_parquet(covariates_parquet_path)

    return data_parquet_path, covariates_parquet_path

@pytest.fixture(scope="module")
def stage_1(test_base_dir, create_dummy_data):
    data_parquet_path, covariates_parquet_path = create_dummy_data
    
    stage_1 = DummyStage(
        name="stage_1",
        directory=test_base_dir / "stage_1",
        config=StageConfig(),
        input_validation=dict(
            data=Data(
                stage="stage_1",
                path=data_parquet_path,
                format="parquet",
                shape=(1, 2),
                columns=dict(
                    id_col=ColumnSpec(type=int),
                    bounded_col=ColumnSpec(
                        type=float,
                        constraints=[
                            Constraint(name="bounds", args=dict(ge=0, le=1))
                        ]
                    ),
                    str_col=ColumnSpec(
                        type=str,
                        constraints=[
                            Constraint(name="is_in", args=dict(other={"a", "b", "c"}))
                        ]
                    )
                )
            ),
            covariates=Data(
                stage="stage_1",
                path=covariates_parquet_path,
                format="parquet",
                columns=dict(
                    id_col=ColumnSpec(type=int),
                    str_col=ColumnSpec(
                        type=str,
                        constraints=[
                            Constraint(name="is_in", args=dict(other=["cov1", "cov2", "cov3"]))
                        ]
                    )
                )
            )
        ),
        output_validation=dict(
            predictions=Data(
                stage="stage_1",
                path=test_base_dir / "stage_1" / "predictions.parquet",
                format="parquet",
                columns=dict(
                    id_col=ColumnSpec(type=int),
                    prediction_col=ColumnSpec(
                        type=float,
                        constraints=[
                            Constraint(name="bounds", args=dict(ge=-1, le=1))
                        ]
                    )
                )
            )
        )
    )
    stage_1.directory = test_base_dir / "stage_1"
    stage_1(
        data=data_parquet_path,
        covariates=covariates_parquet_path
    )
    
    return stage_1

@pytest.fixture(scope="module")
def pipeline_with_single_stage(test_base_dir, stage_1):
    """A sample pipeline with a single stage and no dependencies."""
    pipeline = Pipeline(
        name="test_pipeline",
        config=PipelineConfig(ids=["age_group_id", "location_id"]),
        directory=test_base_dir,
        data=test_base_dir / "data" / "data.parquet",
        groupby=["age_group_id"]
    )
    pipeline.add_stage(stage_1)
    
    return pipeline

@pytest.mark.integration
def test_pipeline_build_single_stage(test_base_dir, pipeline_with_single_stage):
    """Test building a pipeline with a single stage and no dependencies."""
    pipeline_with_single_stage.build()
    
    with open(test_base_dir / f"{pipeline_with_single_stage.name}.json", "r") as f:
        pipeline_dict_actual = json.load(f)
    
    pipeline_dict_expected = {
        "name": "test_pipeline",
        "directory": str(test_base_dir),
        "data": str(test_base_dir / "data" / "data.parquet"),
        "groupby": ["age_group_id"],
        "config": {
            "ids": ["age_group_id", "location_id"],
            "obs": "obs",
            "pred": "pred",
            "weights": "weights",
            "test": "test",
            "holdouts": [],
            "mtype": "binomial"
        },
        "stages": {
            "stage_1": {
                "name": "stage_1",
                "type": "DummyStage",
                "module": __file__,
                "config": {
                    "ids": ["age_group_id", "location_id"],
                    "obs": "obs",
                    "pred": "pred",
                    "weights": "weights",
                    "test": "test",
                    "holdouts": [],
                    "mtype": "binomial"
                },
                "input": {
                    "data": str(test_base_dir / "data" / "data.parquet"),
                    "covariates": str(test_base_dir / "data" / "covariates.parquet")
                },
                "input_validation": {
                    "data": {
                        "stage": "stage_1",
                        "path": str(test_base_dir / "data" / "data.parquet"),
                        "format": "parquet",
                        "shape": [1, 2],
                        "columns": {
                            "id_col": {
                                "type": "int",
                                "constraints": None
                            },
                            "bounded_col": {
                                "type": "float",
                                "constraints": [
                                    {
                                        "name": "bounds",
                                        "args": {
                                            "ge": 0,
                                            "le": 1
                                        }
                                    }
                                ]
                            },
                            "str_col": {
                                "type": "str",
                                "constraints": [
                                    {
                                        "name": "is_in",
                                        "args": {
                                            "other": ["a", "b", "c"]
                                        }
                                    }
                                ]
                            }
                        }
                    },
                    "covariates": {
                        "stage": "stage_1",
                        "path": str(test_base_dir / "data" / "covariates.parquet"),
                        "format": "parquet",
                        "shape": None,
                        "columns": {
                            "id_col": {
                                "type": "int",
                                "constraints": None
                            },
                            "str_col": {
                                "type": "str",
                                "constraints": [
                                    {
                                        "name": "is_in",
                                        "args": {
                                            "other": ["cov1", "cov2", "cov3"]
                                        }
                                    }
                                ]
                            }
                        }
                    }
                },
                "output_validation": {
                    "predictions": {
                        "stage": "stage_1",
                        "path": str(test_base_dir / "stage_1" / "predictions.parquet"),
                        "format": "parquet",
                        "shape": None,
                        "columns": {
                            "id_col": {
                                "type": "int",
                                "constraints": None
                            },
                            "prediction_col": {
                                "type": "float",
                                "constraints": [
                                    {
                                        "name": "bounds",
                                        "args": {
                                            "ge": -1,
                                            "le": 1
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        },
        "dependencies": {"stage_1": []}
    }
    
    assert_equal_unordered(pipeline_dict_actual, pipeline_dict_expected)
