"""Test stage input/output."""

from pathlib import Path

import pytest

from onemod.constraints import Constraint
from onemod.dtypes import ColumnSpec, Data
from onemod.stage import Stage


class DummyStage(Stage):
    _skip: list[str] = ["fit", "predict"]
    _required_input: dict[str, Data] = {
        "data": Data(format="parquet"),
        "covariates": Data(format="csv"),
    }
    _optional_input: dict[str, Data] = {"priors": Data(format="pkl")}
    _output_items: dict[str, Data] = {
        "predictions": Data(format="parquet"),
        "model": Data(format="pkl"),
    }

    def run(self):
        pass


@pytest.fixture
def stage_1(test_base_dir):
    stage_1 = DummyStage(
        name="stage_1",
        config_path=test_base_dir / "pipeline.json",
        input_validation=dict(
            data=Data(
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
                format="parquet",
                path=test_base_dir / "stage_1" / "predictions.parquet",
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
    stage_1(
        data=test_base_dir / "stage_0" / "data.parquet",
        covariates=test_base_dir / "stage_0" / "covariates.csv",
    )
    return stage_1


@pytest.fixture
def stage_1_model_expected(test_base_dir):
    return {
        "name": "stage_1",
        "config": {},
        "type": "DummyStage",
        "module": Path(__file__),
        "input_validation": {
            "covariates": {
                "stage": None,
                "methods": None,
                "format": "csv",
                "path": None,
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
                "stage": None,
                "methods": None,
                "format": "parquet",
                "path": None,
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
                "methods": None,
                "format": "parquet",
                "path": test_base_dir / "stage_1" / "predictions.parquet",
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
                "format": "parquet",
                "path": test_base_dir / "stage_0" / "data.parquet",
            },
            "covariates": {
                "format": "csv",
                "path": test_base_dir / "stage_0" / "covariates.csv",
            },
        },
        "groupby": None,
        "crossby": None,
    }


@pytest.fixture
def stage_2(test_base_dir, stage_1):
    stage_2 = DummyStage(
        name="stage_2",
        config_path=test_base_dir / "pipeline.json",
        input_validation=dict(
            data=Data(
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
                path=test_base_dir / "stage_2" / "predictions.parquet",
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
    stage_2(
        data=stage_1.output["predictions"],
        covariates=test_base_dir / "stage_0" / "covariates.csv",
    )
    return stage_2


@pytest.fixture
def stage_2_model_expected(test_base_dir):
    return {
        "name": "stage_2",
        "config": {},
        "type": "DummyStage",
        "module": Path(__file__),
        "input_validation": {
            "data": {
                "stage": None,
                "methods": None,
                "format": "parquet",
                "path": None,
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
                "methods": None,
                "format": "parquet",
                "path": test_base_dir / "stage_2" / "predictions.parquet",
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
                "format": "parquet",
                "path": test_base_dir / "stage_1" / "predictions.parquet",
            },
            "covariates": {
                "format": "csv",
                "path": test_base_dir / "stage_0" / "covariates.csv",
            },
        },
        "groupby": None,
        "crossby": None,
    }


@pytest.mark.integration
def test_input_types(test_base_dir, stage_1):
    assert "data" in stage_1.input_validation
    assert stage_1.input_validation["data"].path is None
    assert stage_1.input_validation["data"].format == "parquet"
    assert stage_1.input_validation["data"].shape is None
    assert stage_1.dependencies == []


@pytest.mark.integration
def test_output_types(stage_1):
    assert "predictions" in stage_1.output_validation
    assert (
        stage_1.output_validation["predictions"].path
        == stage_1.dataif.get_path("output") / "predictions.parquet"
    )
    assert stage_1.output_validation["predictions"].format == "parquet"
    assert stage_1.output_validation["predictions"].shape is None


@pytest.mark.integration
def test_stage_model(
    stage_1, stage_1_model_expected, stage_2, stage_2_model_expected
):
    stage_1_model_actual = stage_1.model_dump()
    assert stage_1_model_actual == stage_1_model_expected

    stage_2_model_actual = stage_2.model_dump()
    assert stage_2_model_actual == stage_2_model_expected
