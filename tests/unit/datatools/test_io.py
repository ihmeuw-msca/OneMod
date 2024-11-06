import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from onemod.datatools.io import (
    CSVIO,
    JSONIO,
    TOMLIO,
    YAMLIO,
    ParquetIO,
    PickleIO,
)

tmpdir = Path(__file__).parents[1] / "tmp"


@pytest.fixture(scope="class", autouse=True)
def rm_tmpdir_after_tests():
    yield
    if tmpdir.exists():
        shutil.rmtree(tmpdir)


@pytest.fixture
def data():
    return {"a": [1, 2, 3], "b": [4, 5, 6]}


def test_csvio(data):
    data = pd.DataFrame(data)
    port = CSVIO()
    port.dump(data, tmpdir / "file.csv")
    loaded_data = port.load(tmpdir / "file.csv")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


def test_jsonio(data):
    port = JSONIO()
    port.dump(data, tmpdir / "file.json")
    loaded_data = port.load(tmpdir / "file.json")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


def test_yamlio(data):
    port = YAMLIO()
    port.dump(data, tmpdir / "file.yaml")
    loaded_data = port.load(tmpdir / "file.yaml")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


def test_parquetio(data):
    data = pd.DataFrame(data)
    port = ParquetIO()
    port.dump(data, tmpdir / "file.parquet")
    loaded_data = port.load(tmpdir / "file.parquet")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


def test_pickleio(data):
    port = PickleIO()
    port.dump(data, tmpdir / "file.pkl")
    loaded_data = port.load(tmpdir / "file.pkl")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


def test_tomlio(data):
    port = TOMLIO()
    port.dump(data, tmpdir / "file.toml")
    loaded_data = port.load(tmpdir / "file.toml")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])
