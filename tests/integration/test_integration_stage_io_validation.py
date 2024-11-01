"""Test stage input/output."""

from pathlib import Path

import pytest

from onemod.config import StageConfig
from onemod.constraints import Constraint
from onemod.dtypes import ColumnSpec, Data
from onemod.stage import Stage


class DummyStage(Stage):
    config: StageConfig
    _required_input: set[str] = {"data.parquet", "covariates.csv"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}

    def run(self):
        pass


@pytest.fixture
def stage_1(test_base_dir):
    stage_1 = DummyStage(
        name="stage_1",
        config={},
        input_validation=dict(
            data=Data(
                stage="stage_0",
                path="data.parquet",
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
                path="covariates.csv",
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
                path="predictions.parquet",
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
    stage_1(
        data=test_base_dir / "stage_0" / "data.parquet",
        covariates=test_base_dir / "stage_0" / "covariates.csv",
    )
    return stage_1


@pytest.fixture
def stage_1_model_expected(test_base_dir):
    return {
        "name": "stage_1",
        "type": "DummyStage",
        "module": Path(__file__),
        "config": {
            "coef_bounds": None,
            "holdout_columns": None,
            "id_columns": None,
            "model_type": None,
            "observation_column": None,
            "prediction_column": None,
            "test_column": None,
            "weights_column": None,
        },
        "input_validation": {
            "covariates": {
                "stage": "stage_0",
                "path": "covariates.csv",
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
                "path": "data.parquet",
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
                "path": "predictions.parquet",
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
            "data": str(test_base_dir / "stage_0" / "data.parquet"),
            "covariates": str(test_base_dir / "stage_0" / "covariates.csv"),
        },
    }


@pytest.fixture
def stage_2(test_base_dir, stage_1):
    stage_2 = DummyStage(
        name="stage_2",
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
                path="predictions.parquet",
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
        "type": "DummyStage",
        "config": {
            "coef_bounds": None,
            "holdout_columns": None,
            "id_columns": None,
            "model_type": None,
            "observation_column": None,
            "prediction_column": None,
            "test_column": None,
            "weights_column": None,
        },
        "module": Path(__file__),
        "input_validation": {
            "data": {
                "stage": "stage_1",
                "path": "predictions.parquet",
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
                "path": "predictions.parquet",
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
                "path": "predictions.parquet",
                "format": "parquet",
                "shape": None,
                "columns": None,
            },
            "covariates": str(test_base_dir / "stage_0" / "covariates.csv"),
        },
    }


@pytest.mark.integration
def test_input_types(test_base_dir, stage_1):
    assert "data" in stage_1.input_validation
    assert stage_1.input_validation["data"].path == Path("data.parquet")
    assert stage_1.input_validation["data"].format == "parquet"
    assert stage_1.input_validation["data"].shape is None
    assert stage_1.dependencies == set()


@pytest.mark.integration
def test_output_types(stage_1):
    assert "predictions" in stage_1.output_validation
    assert stage_1.output_validation["predictions"].path == Path(
        "predictions.parquet"
    )
    assert stage_1.output_validation["predictions"].format == "parquet"
    assert stage_1.output_validation["predictions"].shape is None


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
