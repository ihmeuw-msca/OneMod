"""Test stage input/output."""

import json
from pathlib import Path

import pytest

from onemod.config import Config
from onemod.io import Input, Output
from onemod.stage import Stage
from onemod.dtypes import Data


class DummyStage(Stage):
    config: Config
    _required_input: set[str] = {"data.parquet", "covariates.csv"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}


@pytest.fixture(scope="module")
def stage_1(tmp_path_factory):
    stage_1 = DummyStage(name="stage_1", config={})
    stage_1.directory = tmp_path_factory.mktemp("example") / stage_1.name
    stage_1(data="/path/to/data.parquet", covariates="/path/to/covariates.csv")
    return stage_1


@pytest.fixture(scope="module")
def stage_2(stage_1):
    stage_2 = DummyStage(name="stage_2", config={})
    stage_2.directory = stage_1.directory.parent / stage_2.name
    stage_2(
        data=stage_1.output["predictions"], covariates="/path/to/covariates.csv"
    )
    return stage_2


@pytest.mark.integration
def test_input(stage_1):
    assert stage_1.input == Input(
        stage=stage_1.name,
        items={
            "data": Path("/path/to/data.parquet"),
            "covariates": Path("/path/to/covariates.csv"),
        },
        required=stage_1._required_input,
        optional=stage_1._optional_input,
    )
    assert stage_1.dependencies == set()


@pytest.mark.integration
def test_output(stage_1):
    assert stage_1.output == Output(
        stage=stage_1.name,
        items={
            "predictions": Data(
                stage=stage_1.name,
                path=stage_1.directory / "predictions.parquet",
            ),
            "model": Data(
                stage=stage_1.name, path=stage_1.directory / "model.pkl"
            ),
        },
    )


@pytest.mark.integration
def test_input_with_dependency(stage_1, stage_2):
    assert stage_2.input == Input(
        stage=stage_2.name,
        items={
            "data": Data(
                stage=stage_1.name,
                path=stage_1.directory / "predictions.parquet",
            ),
            "covariates": Path("/path/to/covariates.csv"),
        },
        required=stage_1._required_input,
        optional=stage_1._optional_input,
    )
    assert stage_2.dependencies == {"stage_1"}


@pytest.mark.unit
def test_input_with_missing():
    stage_3 = DummyStage(name="stage_3", config={})
    with pytest.raises(KeyError) as error:
        stage_3(priors="/path/to/priors.pkl")
    observed = str(error.value).strip('"')
    expected = f"{stage_3.name} missing required input: "
    assert (
        observed == expected + "['data', 'covariates']"
        or observed == expected + "['covariates', 'data']"
    )
    assert stage_3.input.items == {}


@pytest.mark.unit
def test_dependencies(stage_1, stage_2):
    stage_3 = DummyStage(name="stage_3", config={})
    assert stage_3.dependencies == set()
    stage_3(
        data=stage_1.output["predictions"],
        covariates="/path/to/covariates.csv",
        priors=stage_2.output["model"],
    )
    assert stage_3.dependencies == {"stage_1", "stage_2"}


@pytest.mark.unit
def test_to_json(stage_1, stage_2):
    stage_2.to_json()
    with open(stage_2.directory / (stage_2.name + ".json"), "r") as f:
        config = json.load(f)
    print(config)
    assert config["input"] == {
        "data": {
            "stage": stage_1.name,
            "path": str(stage_1.output["predictions"].path),
            "format": "parquet",
            "shape": None,
            "columns": None,
        },
        "covariates": "/path/to/covariates.csv",
    }


@pytest.mark.unit
def test_to_json_no_input(tmp_path):
    stage_3 = DummyStage(name="stage_3", config={})
    stage_3.directory = tmp_path
    stage_3.to_json()
    with open(stage_3.directory / (stage_3.name + ".json"), "r") as f:
        config = json.load(f)
    assert "input" not in config


@pytest.mark.unit
def test_from_json(stage_2):
    stage_2.to_json()
    stage_2_new = DummyStage.from_json(
        stage_2.directory / (stage_2.name + ".json")
    )
    assert stage_2_new.input == stage_2.input
    assert stage_2_new.output == stage_2.output


@pytest.mark.unit
def test_stage_model(stage_1, stage_2):
    stage_1_model_actual = stage_1.model_dump()

    stage_1_model_expected = {
        "name": "stage_1",
        "type": "DummyStage",
        "config": {},
        "input_validation": {},
        "output_validation": {},
        "module": __file__,
        "input": {
            "data": "/path/to/data.parquet",
            "covariates": "/path/to/covariates.csv",
        },
    }

    assert stage_1_model_actual == stage_1_model_expected

    stage_2_model_actual = stage_2.model_dump()

    stage_2_model_expected = {
        "name": "stage_2",
        "type": "DummyStage",
        "config": {},
        "input_validation": {},
        "output_validation": {},
        "module": __file__,
        "input": {
            "data": {
                "stage": "stage_1",
                "path": str(stage_1.directory / "predictions.parquet"),
                "format": "parquet",
                "shape": None,
                "columns": None,
            },
            "covariates": "/path/to/covariates.csv",
        },
    }

    assert stage_2_model_actual == stage_2_model_expected
