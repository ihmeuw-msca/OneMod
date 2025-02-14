"""Test stage input/output."""

from pathlib import Path
from typing import Any

import pytest

from onemod.dtypes import Data
from onemod.io import Input, Output
from onemod.stage import Stage


class DummyStage(Stage):
    _skip: list[str] = ["fit", "predict"]
    _required_input: dict[str, dict[str, Any]] = {
        "data": {"format": "parquet"},
        "covariates": {"format": "csv"},
    }
    _optional_input: dict[str, dict[str, Any]] = {"priors": {"format": "pkl"}}
    _output_items: dict[str, dict[str, Any]] = {
        "predictions": {"format": "parquet"},
        "model": {"format": "pkl"},
    }

    def run(self):
        pass


@pytest.fixture
def stage_1(tmp_path):
    stage_1 = DummyStage(
        name="stage_1", config_path=tmp_path / "dummy_pipeline.json"
    )
    stage_1(data="/path/to/data.parquet", covariates="/path/to/covariates.csv")
    return stage_1


@pytest.fixture
def stage_2(stage_1):
    stage_2 = DummyStage(
        name="stage_2", config_path=stage_1.dataif.get_path("config")
    )
    stage_2(
        data=stage_1.output["predictions"], covariates="/path/to/covariates.csv"
    )
    return stage_2


@pytest.mark.integration
def test_input(stage_1):
    assert stage_1.input == Input(
        stage=stage_1.name,
        required=stage_1._required_input,
        optional=stage_1._optional_input,
        items={
            "data": Input.path_to_data("/path/to/data.parquet"),
            "covariates": Input.path_to_data("/path/to/covariates.csv"),
        },
    )
    assert stage_1.dependencies == []


@pytest.mark.integration
def test_output(stage_1):
    assert stage_1.output == Output(
        stage=stage_1.name,
        directory=stage_1.dataif.get_path("output"),
        items={
            "predictions": Data(stage=stage_1.name, format="parquet"),
            "model": Data(stage=stage_1.name, format="pkl"),
        },
    )


@pytest.mark.integration
def test_input_with_dependency(stage_1, stage_2):
    stage_1_output = stage_1.dataif.get_path("output")
    assert stage_2.input == Input(
        stage=stage_2.name,
        required=stage_1._required_input,
        optional=stage_1._optional_input,
        items={
            "data": Data(
                stage=stage_1.name, path=stage_1_output / "predictions.parquet"
            ),
            "covariates": Input.path_to_data("/path/to/covariates.csv"),
        },
    )
    assert stage_2.dependencies == ["stage_1"]


@pytest.mark.unit
def test_dependencies(stage_1, stage_2):
    stage_3 = DummyStage(
        name="stage_3", config_path=stage_1.dataif.get_path(key="config")
    )
    assert stage_3.dependencies == []
    stage_3(
        data=stage_1.output["predictions"],
        covariates="/path/to/covariates.csv",
        priors=stage_2.output["model"],
    )
    assert stage_3.dependencies == ["stage_1", "stage_2"]


@pytest.mark.unit
def test_stage_model(stage_1, stage_2):
    stage_1_model_actual = stage_1.model_dump()

    stage_1_model_expected = {
        "name": "stage_1",
        "type": "DummyStage",
        "config": {},
        "input_validation": None,
        "output_validation": None,
        "module": Path(__file__),
        "input": {
            "data": {
                "stage": None,
                "methods": None,
                "format": "parquet",
                "path": Path("/path/to/data.parquet"),
                "shape": None,
                "columns": None,
            },
            "covariates": {
                "stage": None,
                "methods": None,
                "format": "csv",
                "path": Path("/path/to/covariates.csv"),
                "shape": None,
                "columns": None,
            },
        },
        "groupby": None,
        "crossby": None,
    }

    assert stage_1_model_actual == stage_1_model_expected

    stage_2_model_actual = stage_2.model_dump()

    stage_2_model_expected = {
        "name": "stage_2",
        "type": "DummyStage",
        "config": {},
        "input_validation": None,
        "output_validation": None,
        "module": Path(__file__),
        "input": {
            "data": {
                "stage": "stage_1",
                "methods": None,
                "path": stage_1.dataif.get_path("output")
                / "predictions.parquet",
                "format": "parquet",
                "shape": None,
                "columns": None,
            },
            "covariates": {
                "stage": None,
                "methods": None,
                "path": Path("/path/to/covariates.csv"),
                "format": "csv",
                "shape": None,
                "columns": None,
            },
        },
        "groupby": None,
        "crossby": None,
    }

    assert stage_2_model_actual == stage_2_model_expected
