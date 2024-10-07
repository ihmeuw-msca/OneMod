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


TEST_OUTPUT = Output(
    stage="test_stage",
    items={
        "predictions": {
            "stage": "test_stage",
            "path": "/path/to/predictions.parquet",
        }
    },
)


def test_serialize():
    assert TEST_OUTPUT.model_dump() == TEST_OUTPUT.items


def test_get():
    assert TEST_OUTPUT.get("predictions") == Data(
        stage="test_stage", path=Path("/path/to/predictions.parquet")
    )
    assert TEST_OUTPUT.get("dummy") is None
    assert TEST_OUTPUT.get("dummy", "default") == "default"


def test_getitem():
    assert TEST_OUTPUT["predictions"] == Data(
        stage="test_stage", path=Path("/path/to/predictions.parquet")
    )
    with pytest.raises(KeyError) as error:
        TEST_OUTPUT["dummy"]
    print(error.value)
    print(str(error.value))
    assert (
        str(error.value).strip('"')
        == "test_stage does not contain output 'dummy'"
    )


def test_contains():
    assert "predictions" in TEST_OUTPUT
    assert "dummy" not in TEST_OUTPUT


def test_frozen():
    with pytest.raises(ValidationError):
        TEST_OUTPUT.items = {}


def test_no_setitem():
    with pytest.raises(TypeError):
        TEST_OUTPUT["key"] = "value"
