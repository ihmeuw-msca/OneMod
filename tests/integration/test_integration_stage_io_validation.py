"""Test stage input/output."""

import json
from pathlib import Path

import pytest

from onemod.config import Config
from onemod.constraints import Constraint
from onemod.io import Input, Output
from onemod.stage import Stage
from onemod.types import Data


class DummyStage(Stage):
    config: Config
    _required_input: set[str] = {"data.parquet", "covariates.csv"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}

@pytest.fixture(scope="module")
def example_base_dir(tmp_path_factory):
    example_base_dir = tmp_path_factory.mktemp("example")
    return example_base_dir

@pytest.fixture(scope="module")
def stage_1(example_base_dir):
    stage_1 = DummyStage(
        name="stage_1",
        directory=example_base_dir / "stage_1",
        config={},
        input_types=dict(
            data=Data(
                stage="stage_1",
                path=example_base_dir / "stage_1" / "data.parquet",
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
                path=example_base_dir / "stage_1" / "covariates.csv",
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
                path=example_base_dir / "stage_1" / "predictions.parquet",
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
    stage_1.directory = example_base_dir / "stage_1"
    stage_1(
        data=example_base_dir / "stage_1" / "data.parquet",
        covariates=example_base_dir / "stage_1" / "covariates.csv"
    )
    return stage_1


@pytest.fixture(scope="module")
def stage_2(example_base_dir, stage_1):
    stage_2 = DummyStage(
        name="stage_2",
        directory=example_base_dir / "stage_2",
        config={},
        input_types=dict(
            data=Data(
                stage="stage_2",
                path=stage_1.output["predictions"].path,
                format="parquet",
                columns=dict(
                    id_col=dict(type=int),
                    prediction_col=dict(
                        type=float,
                        constraints=[
                            Constraint("bounds", ge=0, le=1)
                        ]
                    )
                )
            )
        ),
        output_types=dict(
            data=Data(
                stage="stage_2",
                path=example_base_dir / "stage_2" / "predictions.parquet",
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
    stage_2.directory = example_base_dir / "stage_2"
    stage_2(
        data=stage_1.output["predictions"],
        covariates="/path/to/covariates.csv"
    )
    return stage_2

@pytest.mark.integration
def test_input_types(stage_1):
    assert "data" in stage_1.input_types
    assert stage_1.input_types["data"].path == stage_1.directory / "data.parquet"
    assert stage_1.input_types["data"].format == "parquet"
    assert stage_1.input_types["data"].shape == None
    assert stage_1.dependencies == set()


@pytest.mark.integration
def test_output_types(stage_1):
    assert "predictions" in stage_1.output_types
    assert stage_1.output_types["predictions"].path == stage_1.directory / "predictions.parquet"
    assert stage_1.output_types["predictions"].format == "parquet"
    assert stage_1.output_types["predictions"].shape == None

@pytest.mark.integration
def test_to_dict(stage_1, stage_2):
    print("model_dump_json")
    print(stage_2.model_dump_json())
    print("to_dict")
    print(stage_2.to_dict())
    stage_2.to_dict()
    with open(stage_2.directory / (stage_2.name + ".json"), "r") as f:
        stage_2_loaded = json.load(f)
    assert stage_2_loaded["input"] == {
        "data": {
            "stage": stage_1.name,
            "path": str(stage_1.output["predictions"].path),
            "format": "parquet",
            "shape": None,
            "columns": {
                "id_col": {"type": "int"},
                "prediction_col": {
                    "type": "float",
                    "constraints": [{"name": "bounds", "args": [0, 1]}]
                }
            }
        },
        "covariates": "/path/to/covariates.csv",
    }

@pytest.mark.integration
def test_from_json(stage_2):
    stage_2.to_json()
    stage_2_new = DummyStage.from_json(
        stage_2.directory / (stage_2.name + ".json")
    )
    print("stage_2_new")
    print(stage_2_new)
    assert stage_2_new.name == stage_2.name
    assert stage_2_new.directory == stage_2.directory
    assert stage_2_new.input == stage_2.input
    assert stage_2_new.output == stage_2.output
    assert stage_2_new.dependencies == stage_2.dependencies
    assert stage_2_new._required_input == stage_2._required_input
    assert stage_2_new._optional_input == stage_2._optional_input
    assert stage_2_new.input_types == stage_2.input_types
    assert stage_2_new.output_types == stage_2.output_types
    assert stage_2_new.config == stage_2.config
