import numpy as np
import pandas as pd
import polars as pl
import pytest

from onemod.fsutils.io import CSVIO, JSONIO, TOMLIO, YAMLIO, ParquetIO, PickleIO


@pytest.fixture
def data():
    return {"a": [1, 2, 3], "b": [4, 5, 6]}


@pytest.mark.unit
def test_csvio_eager(data, tmp_path):
    data = pd.DataFrame(data)
    port = CSVIO()
    port.dump(data, tmp_path / "file.csv")
    loaded_data = port.load_eager(tmp_path / "file.csv")

    assert type(loaded_data) is pd.DataFrame

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


@pytest.mark.unit
def test_csvio_lazy(data, tmp_path):
    data = pd.DataFrame(data)
    port = CSVIO()
    port.dump(data, tmp_path / "file.csv")
    lazy_loaded_data = port.load_lazy(tmp_path / "file.csv")

    assert type(lazy_loaded_data) is pl.LazyFrame

    loaded_data = lazy_loaded_data.collect()

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


@pytest.mark.unit
def test_jsonio(data, tmp_path):
    port = JSONIO()
    port.dump(data, tmp_path / "file.json")
    loaded_data = port.load(tmp_path / "file.json")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


@pytest.mark.unit
def test_yamlio(data, tmp_path):
    port = YAMLIO()
    port.dump(data, tmp_path / "file.yaml")
    loaded_data = port.load(tmp_path / "file.yaml")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


@pytest.mark.unit
def test_parquetio_eager_pandas(data, tmp_path):
    data = pd.DataFrame(data)
    port = ParquetIO()
    port.dump(data, tmp_path / "file.parquet")
    loaded_data = port.load_eager(tmp_path / "file.parquet")

    assert type(loaded_data) is pd.DataFrame

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


@pytest.mark.unit
def test_parquetio_eager_polars(data, tmp_path):
    data = pl.DataFrame(data)
    port = ParquetIO()
    port.dump(data, tmp_path / "file.parquet")
    loaded_data = port.load_eager(tmp_path / "file.parquet", backend="polars")

    assert type(loaded_data) is pl.DataFrame

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


@pytest.mark.unit
def test_parquetio_lazy(data, tmp_path):
    data = pd.DataFrame(data)
    port = ParquetIO()
    port.dump(data, tmp_path / "file.parquet")
    lazy_loaded_data = port.load_lazy(tmp_path / "file.parquet")

    assert type(lazy_loaded_data) is pl.LazyFrame

    loaded_data = lazy_loaded_data.collect()

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


@pytest.mark.unit
def test_pickleio(data, tmp_path):
    port = PickleIO()
    port.dump(data, tmp_path / "file.pkl")
    loaded_data = port.load(tmp_path / "file.pkl")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])


@pytest.mark.unit
def test_tomlio(data, tmp_path):
    port = TOMLIO()
    port.dump(data, tmp_path / "file.toml")
    loaded_data = port.load(tmp_path / "file.toml")

    for key in ["a", "b"]:
        assert np.allclose(data[key], loaded_data[key])
