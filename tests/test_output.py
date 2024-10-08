"""Test output class.

Notes
-----
Output objects are frozen and don't have a __setitem__ method, but there
is currently nothing to prevent a user from setting items with
`output.items[key] = `value`.

"""

from pathlib import Path

from pydantic import ValidationError
import pytest

from onemod.io import Data, Output


ITEMS = {
    "predictions": {"stage": "stage", "path": "/path/to/predictions.parquet"}
}
OUTPUT = Output(stage="stage", items=ITEMS)


def test_serialize():
    assert OUTPUT.model_dump() == ITEMS


def test_get():
    assert OUTPUT.get("predictions") == Data(
        stage="stage", path=Path("/path/to/predictions.parquet")
    )
    assert OUTPUT.get("dummy") is None
    assert OUTPUT.get("dummy", "default") == "default"


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


def test_contains():
    assert "predictions" in OUTPUT
    assert "dummy" not in OUTPUT


def test_frozen():
    with pytest.raises(ValidationError):
        OUTPUT.items = {}


def test_no_setitem():
    with pytest.raises(TypeError):
        OUTPUT["item_name"] = "item_value"
