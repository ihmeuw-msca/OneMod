"""Test stage input/output."""

import json
from pathlib import Path

import pytest

from onemod.config import Config
from onemod.io import Data, Input, Output
from onemod.stage import Stage


class DummyStage(Stage):
    config: Config
    _required_input: set[str] = {"data.parquet", "covariates.csv"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}


@pytest.fixture(scope="module")
def stage_1(tmp_path_factory):
    stage_1 = DummyStage(
        name="stage_1",
        config={},
        input={
            "data": "/path/to/data.parquet",
            "covariates": "/path/to/covariates.csv",
        },
    )
    stage_1.directory = tmp_path_factory.mktemp("example") / stage_1.name
    return stage_1


@pytest.fixture(scope="module")
def stage_2(stage_1):
    stage_2 = DummyStage(name="stage_2", config={})
    stage_2.directory = stage_1.directory.parent / stage_2.name
    stage_2(
        data=stage_1.output["predictions"], covariates="/path/to/covariates.csv"
    )
    return stage_2


def test_input_from_init(stage_1):
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


def test_input_from_call(stage_1, stage_2):
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


def test_input_from_call_missing():
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


def test_dependencies(stage_1, stage_2):
    stage_3 = DummyStage(name="stage_3", config={})
    assert stage_3.dependencies == set()
    stage_3(
        data=stage_1.output["predictions"],
        covariates="/path/to/covariates.csv",
        priors=stage_2.output["model"],
    )
    assert stage_3.dependencies == {"stage_1", "stage_2"}


def test_to_json(stage_1, stage_2):
    stage_2.to_json()
    with open(stage_2.directory / (stage_2.name + ".json"), "r") as f:
        config = json.load(f)
    assert config["input"] == {
        "data": {
            "stage": stage_1.name,
            "path": str(stage_1.output["predictions"].path),
        },
        "covariates": "/path/to/covariates.csv",
    }
    assert "output" not in config


def test_to_json_no_input(tmp_path):
    stage_3 = DummyStage(name="stage_3", config={})
    stage_3.directory = tmp_path
    stage_3.to_json()
    with open(stage_3.directory / (stage_3.name + ".json"), "r") as f:
        config = json.load(f)
    assert config["input"] is None  # FIXME: wanted to exclude if None
    assert "output" not in config


def test_from_json(stage_2):
    stage_2.to_json()
    stage_2_new = DummyStage.from_json(
        stage_2.directory / (stage_2.name + ".json")
    )
    assert stage_2_new.input == stage_2.input
    assert stage_2_new.output == stage_2.output
