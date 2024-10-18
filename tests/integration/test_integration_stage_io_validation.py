"""Test stage input/output."""

import json
from pathlib import Path

import pytest

from onemod.config import Config
from onemod.constraints import Constraint
from onemod.stage import Stage
from onemod.dtypes import ColumnSpec, Data


class DummyStage(Stage):
    config: Config
    _required_input: set[str] = {"data.parquet", "covariates.csv"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}


@pytest.fixture
def example_base_dir(tmp_path_factory):
    example_base_dir = tmp_path_factory.mktemp("example")
    return example_base_dir


@pytest.fixture
def stage_1(example_base_dir):
    stage_1 = DummyStage(
        name="stage_1",
        directory=example_base_dir / "stage_1",
        config={},
        input_validation=dict(
            data=Data(
                stage="stage_0",
                path=example_base_dir / "stage_0" / "data.parquet",
                format="parquet",
                columns=dict(
                    id_col=ColumnSpec(type=int),
                    bounded_col=ColumnSpec(
                        type=float,
                        constraints=[
                            Constraint(name="bounds", args=dict(ge=0, le=1))
                        ],
                    ),
                    str_col=ColumnSpec(
                        type=str,
                        constraints=[
                            Constraint(
                                name="is_in", args=dict(other=["a", "b", "c"])
                            )
                        ],
                    ),
                ),
            ),
            covariates=Data(
                stage="stage_0",
                path=example_base_dir / "stage_0" / "covariates.csv",
                format="csv",
                columns=dict(
                    id_col=ColumnSpec(type=int),
                    str_col=ColumnSpec(
                        type=str,
                        constraints=[
                            Constraint(
                                name="is_in",
                                args=dict(other=["cov1", "cov2", "cov3"]),
                            )
                        ],
                    ),
                ),
            ),
        ),
        output_validation=dict(
            predictions=Data(
                stage="stage_1",
                path=example_base_dir / "stage_1" / "predictions.parquet",
                format="parquet",
                columns=dict(
                    id_col=ColumnSpec(type=int),
                    prediction_col=ColumnSpec(
                        type=float,
                        constraints=[
                            Constraint(name="bounds", args=dict(ge=-1, le=1))
                        ],
                    ),
                ),
            )
        ),
    )
    stage_1.directory = example_base_dir / "stage_1"
    stage_1(
        data=example_base_dir / "stage_0" / "data.parquet",
        covariates=example_base_dir / "stage_0" / "covariates.csv",
    )
    return stage_1


@pytest.fixture
def stage_1_model_expected(example_base_dir):
    return {
        "name": "stage_1",
        "type": "DummyStage",
        "module": __file__,
        "config": {},
        "input_validation": {
            "covariates": {
                "stage": "stage_0",
                "path": str(example_base_dir / "stage_0" / "covariates.csv"),
                "format": "csv",
                "shape": None,
                "columns": {
                    "id_col": {"type": "int", "constraints": None},
                    "str_col": {
                        "type": "str",
                        "constraints": [
                            {
                                "name": "is_in",
                                "args": {"other": ["cov1", "cov2", "cov3"]},
                            }
                        ],
                    },
                },
            },
            "data": {
                "stage": "stage_0",
                "path": str(example_base_dir / "stage_0" / "data.parquet"),
                "format": "parquet",
                "shape": None,
                "columns": {
                    "id_col": {"type": "int", "constraints": None},
                    "bounded_col": {
                        "type": "float",
                        "constraints": [
                            {"name": "bounds", "args": {"ge": 0, "le": 1}}
                        ],
                    },
                    "str_col": {
                        "type": "str",
                        "constraints": [
                            {
                                "name": "is_in",
                                "args": {"other": ["a", "b", "c"]},
                            }
                        ],
                    },
                },
            },
        },
        "output_validation": {
            "predictions": {
                "stage": "stage_1",
                "path": str(
                    example_base_dir / "stage_1" / "predictions.parquet"
                ),
                "format": "parquet",
                "shape": None,
                "columns": {
                    "id_col": {"type": "int", "constraints": None},
                    "prediction_col": {
                        "type": "float",
                        "constraints": [
                            {"name": "bounds", "args": {"ge": -1, "le": 1}}
                        ],
                    },
                },
            }
        },
        "input": {
            "data": str(example_base_dir / "stage_0" / "data.parquet"),
            "covariates": str(example_base_dir / "stage_0" / "covariates.csv"),
        },
    }


@pytest.fixture
def stage_2(example_base_dir, stage_1):
    stage_2 = DummyStage(
        name="stage_2",
        directory=example_base_dir / "stage_2",
        config={},
        input_validation=dict(
            data=Data(
                stage="stage_1",
                path=stage_1.output["predictions"].path,
                format="parquet",
                columns=dict(
                    id_col=ColumnSpec(type=int),
                    prediction_col=ColumnSpec(
                        type=float,
                        constraints=[
                            Constraint(name="bounds", args=dict(ge=0, le=1))
                        ],
                    ),
                ),
            )
        ),
        output_validation=dict(
            data=Data(
                stage="stage_2",
                path=example_base_dir / "stage_2" / "predictions.parquet",
                format="parquet",
                columns=dict(
                    id_col=ColumnSpec(type=int),
                    prediction_col=ColumnSpec(
                        type=float,
                        constraints=[
                            Constraint(name="bounds", args=dict(ge=-1, le=1))
                        ],
                    ),
                ),
            )
        ),
    )
    stage_2.directory = example_base_dir / "stage_2"
    stage_2(
        data=stage_1.output["predictions"],
        covariates=example_base_dir / "stage_0" / "covariates.csv",
    )
    return stage_2


@pytest.fixture
def stage_2_model_expected(example_base_dir):
    return {
        "name": "stage_2",
        "type": "DummyStage",
        "config": {},
        "module": __file__,
        "input_validation": {
            "data": {
                "stage": "stage_1",
                "path": str(
                    example_base_dir / "stage_1" / "predictions.parquet"
                ),
                "format": "parquet",
                "shape": None,
                "columns": {
                    "id_col": {"type": "int", "constraints": None},
                    "prediction_col": {
                        "type": "float",
                        "constraints": [
                            {"name": "bounds", "args": {"ge": 0, "le": 1}}
                        ],
                    },
                },
            }
        },
        "output_validation": {
            "data": {
                "stage": "stage_2",
                "path": str(
                    example_base_dir / "stage_2" / "predictions.parquet"
                ),
                "format": "parquet",
                "shape": None,
                "columns": {
                    "id_col": {"type": "int", "constraints": None},
                    "prediction_col": {
                        "type": "float",
                        "constraints": [
                            {"name": "bounds", "args": {"ge": -1, "le": 1}}
                        ],
                    },
                },
            }
        },
        "input": {
            "data": {
                "stage": "stage_1",
                "path": str(
                    example_base_dir / "stage_1" / "predictions.parquet"
                ),
                "format": "parquet",
                "shape": None,
                "columns": None,
            },
            "covariates": str(example_base_dir / "stage_0" / "covariates.csv"),
        },
    }


@pytest.mark.integration
def test_input_types(example_base_dir, stage_1):
    assert "data" in stage_1.input_validation
    assert (
        stage_1.input_validation["data"].path
        == example_base_dir / "stage_0" / "data.parquet"
    )
    assert stage_1.input_validation["data"].format == "parquet"
    assert stage_1.input_validation["data"].shape == None
    assert stage_1.dependencies == set()


@pytest.mark.integration
def test_output_types(stage_1):
    assert "predictions" in stage_1.output_validation
    assert (
        stage_1.output_validation["predictions"].path
        == stage_1.directory / "predictions.parquet"
    )
    assert stage_1.output_validation["predictions"].format == "parquet"
    assert stage_1.output_validation["predictions"].shape == None


@pytest.mark.integration
def test_stage_model(
    stage_1, stage_1_model_expected, stage_2, stage_2_model_expected
):
    stage_1_model_actual = stage_1.model_dump()
    print(stage_1_model_actual)
    assert stage_1_model_actual == stage_1_model_expected

    stage_2_model_actual = stage_2.model_dump()
    print(stage_2_model_actual)
    assert stage_2_model_actual == stage_2_model_expected


@pytest.mark.integration
def test_to_json(
    stage_1, stage_1_model_expected, stage_2, stage_2_model_expected
):
    stage_1.to_json(config_path=stage_1.directory / (stage_1.name + ".json"))
    stage_1_loaded_actual = DummyStage.from_json(
        stage_1.directory / (stage_1.name + ".json")
    )
    stage_1_loaded_actual = stage_1_loaded_actual.model_dump()
    assert stage_1_loaded_actual == stage_1_model_expected

    stage_2.to_json(config_path=stage_2.directory / (stage_2.name + ".json"))
    stage_2_loaded_actual = DummyStage.from_json(
        stage_2.directory / (stage_2.name + ".json")
    )
    stage_2_loaded_actual = stage_2_loaded_actual.model_dump()

    assert stage_2_loaded_actual == stage_2_model_expected
