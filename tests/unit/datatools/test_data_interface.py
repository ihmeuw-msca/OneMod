import numpy as np
import pytest
from polars import DataFrame

from onemod.datatools.data_interface import DataInterface


@pytest.fixture
def data():
    return {"a": [1, 2, 3], "b": [4, 5, 6]}


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_data_interface(data, extension, tmp_path):
    dataif = DataInterface(tmp=tmp_path)

    data = DataFrame(data)

    dataif.dump(data, "data" + extension, key="tmp")

    loaded_data = dataif.load("data" + extension, key="tmp")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


def test_add_dir(tmp_path):
    dataif = DataInterface()

    assert len(dataif.paths) == 0

    dataif.add_path("tmp", tmp_path)

    assert len(dataif.paths) == 1
    assert "tmp" in dataif.paths
    assert dataif.paths["tmp"] == tmp_path


def test_add_dir_exist_ok(tmp_path):
    dataif = DataInterface(tmp=tmp_path)

    with pytest.raises(ValueError):
        dataif.add_path("tmp", tmp_path)

    dataif.add_path("tmp", tmp_path, exist_ok=True)


def test_remove_dir(tmp_path):
    dataif = DataInterface(tmp=tmp_path)

    assert len(dataif.paths) == 1

    dataif.remove_path("tmp")

    assert len(dataif.paths) == 0
