"""Test output class."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from onemod.dtypes import Data
from onemod.io import Output

ITEMS = {
    "predictions": {
        "stage": "stage",
        "methods": ["run", "fit"],
        "format": "parquet",
        "path": Path("/path/to/stage/output/predictions.parquet"),
    }
}
OUTPUT = Output(
    stage="stage",
    directory="/path/to/stage/output",
    items={"predictions": Data(format="parquet", methods=["run", "fit"])},
)


@pytest.mark.unit
def test_serialize():
    assert OUTPUT.model_dump() == ITEMS


@pytest.mark.unit
def test_get():
    assert OUTPUT.get("predictions") == Data(**ITEMS["predictions"])
    assert OUTPUT.get("dummy") is None
    assert OUTPUT.get("dummy", "default") == "default"


@pytest.mark.unit
def test_getitem():
    assert OUTPUT["predictions"] == Data(**ITEMS["predictions"])
    with pytest.raises(KeyError) as error:
        OUTPUT["dummy"]
    assert (
        str(error.value).strip('"')
        == "Stage 'stage' does not contain output 'dummy'"
    )


@pytest.mark.unit
def test_contains():
    assert "predictions" in OUTPUT
    assert "dummy" not in OUTPUT


@pytest.mark.unit
def test_frozen():
    with pytest.raises(ValidationError):
        OUTPUT.items = {}


@pytest.mark.unit
def test_no_setitem():
    with pytest.raises(TypeError):
        OUTPUT["item_name"] = "item_value"
