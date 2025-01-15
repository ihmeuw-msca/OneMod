"""Test stage input/output."""

from pathlib import Path

import pytest

from onemod.config import StageConfig
from onemod.dtypes import Data, UniqueList
from onemod.io import Input, Output
from onemod.stage import Stage


class DummyStage(Stage):
    config: StageConfig
    _required_input: UniqueList[str] = ["data.parquet", "covariates.csv"]
    _optional_input: UniqueList[str] = ["priors.pkl"]
    _output: UniqueList[str] = ["predictions.parquet", "model.pkl"]

    def run(self):
        pass


@pytest.fixture
def stage_1():
    stage_1 = DummyStage(name="stage_1", config={})
    stage_1(data="/path/to/data.parquet", covariates="/path/to/covariates.csv")
    return stage_1


@pytest.fixture
def stage_2(stage_1):
    stage_2 = DummyStage(name="stage_2", config={})
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
    assert stage_1.dependencies == []


@pytest.mark.integration
def test_output(stage_1):
    # print(stage_1.output)
    # print(Output(
    #     stage=stage_1.name,
    #     items={
    #         "model": Data(stage=stage_1.name, path="model.pkl"),
    #         "predictions": Data(stage=stage_1.name, path="predictions.parquet"),
    #     },
    # ))
    assert (
        stage_1.output
        == Output(
            stage=stage_1.name,
            items={
                "predictions": Data(
                    stage=stage_1.name,
                    path="predictions.parquet",
                    format="parquet",
                ),
                "model": Data(
                    stage=stage_1.name, path="model.pkl", format="pkl"
                ),  # FIXME: implicit format pending update of Data class with new version of DataInterface
            },
        )
    )


@pytest.mark.integration
def test_input_with_dependency(stage_1, stage_2):
    assert stage_2.input == Input(
        stage=stage_2.name,
        items={
            "data": Data(stage=stage_1.name, path="predictions.parquet"),
            "covariates": Path("/path/to/covariates.csv"),
        },
        required=stage_1._required_input,
        optional=stage_1._optional_input,
    )
    assert stage_2.dependencies == ["stage_1"]


@pytest.mark.unit
def test_input_with_missing():
    stage_3 = DummyStage(name="stage_3", config={})
    with pytest.raises(KeyError) as error:
        stage_3(priors="/path/to/priors.pkl")
    observed = str(error.value).strip('"')
    expected = f"Stage '{stage_3.name}' missing required input: "
    assert (
        observed == expected + "['data', 'covariates']"
        or observed == expected + "['covariates', 'data']"
    )
    assert stage_3.input.items == {}


@pytest.mark.unit
def test_dependencies(stage_1, stage_2):
    stage_3 = DummyStage(name="stage_3", config={})
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
        "input_validation": {},
        "output_validation": {},
        "module": Path(__file__),
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
        "module": Path(__file__),
        "input": {
            "data": {
                "stage": "stage_1",
                "path": "predictions.parquet",
                "format": "parquet",
                "shape": None,
                "columns": None,
            },
            "covariates": "/path/to/covariates.csv",
        },
    }

    assert stage_2_model_actual == stage_2_model_expected
