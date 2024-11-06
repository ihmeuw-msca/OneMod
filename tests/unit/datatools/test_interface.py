import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from onemod.datatools.interface import DataInterface

tmpdir = Path(__file__).parents[1] / "tmp"


@pytest.fixture
def data():
    return {"a": [1, 2, 3], "b": [4, 5, 6]}


@pytest.fixture(scope="class", autouse=True)
def rm_tmpdir_after_tests():
    yield
    if tmpdir.exists():
        shutil.rmtree(tmpdir)


@pytest.mark.parametrize(
    "fextn", [".json", ".yaml", ".pkl", ".csv", ".parquet"]
)
def test_data_interface(data, fextn):
    dataif = DataInterface(tmp=tmpdir)
    if fextn in [".csv", ".parquet"]:
        data = pd.DataFrame(data)
    dataif.dump_tmp(data, "data" + fextn)
    loaded_data = dataif.load_tmp("data" + fextn)

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


def test_add_dir():
    dataif = DataInterface()
    assert len(dataif.keys) == 0
    dataif.add_dir("tmp", tmpdir)
    assert len(dataif.keys) == 1
    assert hasattr(dataif, "tmp")
    assert hasattr(dataif, "load_tmp")
    assert hasattr(dataif, "dump_tmp")


def test_add_dir_exist_ok():
    dataif = DataInterface(tmp=tmpdir)
    with pytest.raises(ValueError):
        dataif.add_dir("tmp", tmpdir)
    dataif.add_dir("tmp", tmpdir, exist_ok=True)


def test_remove_dir():
    dataif = DataInterface(tmp=tmpdir)
    assert len(dataif.keys) == 1
    dataif.remove_dir("tmp")
    assert len(dataif.keys) == 0
