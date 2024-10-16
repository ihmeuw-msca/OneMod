"""Test output class."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from onemod.io import Output
from onemod.dtypes import Data

ITEMS = {
    "predictions": {
        "stage": "stage",
        "path": "/path/to/predictions.parquet",
        "format": "parquet",
        "shape": None,
        "columns": None,
    }
}
OUTPUT = Output(stage="stage", items=ITEMS)


@pytest.mark.unit
def test_serialize():
    print(OUTPUT.model_dump())
    print(ITEMS)
    assert OUTPUT.model_dump() == ITEMS


@pytest.mark.unit
def test_get():
    assert OUTPUT.get("predictions") == Data(
        stage="stage", path=Path("/path/to/predictions.parquet")
    )
    assert OUTPUT.get("dummy") is None
    assert OUTPUT.get("dummy", "default") == "default"


@pytest.mark.unit
def test_getitem():
    assert OUTPUT["predictions"] == Data(
        stage="stage", path=Path("/path/to/predictions.parquet")
    )
    with pytest.raises(KeyError) as error:
        OUTPUT["dummy"]
    print(error.value)
    print(str(error.value))
    assert (
        str(error.value).strip('"') == "stage does not contain output 'dummy'"
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


@pytest.mark.unit
def test_to_dict():
    assert OUTPUT.model_dump() == ITEMS
