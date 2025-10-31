from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from onemod.fsutils.interface import DataInterface


@pytest.fixture
def sample_data1():
    return {"a": [1, 2, 3], "b": [4, 5, 6]}


@pytest.fixture
def sample_data2():
    return {
        "age_group_id": [1, 2, 2, 3],
        "location_id": [10, 20, 20, 30],
        "sex_id": [1, 2, 1, 2],
        "value": [100, 200, 300, 400],
    }


@pytest.fixture
def sample_config():
    return {"param1": "value1", "param2": [1, 2, 3]}


@pytest.mark.unit
@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_data_interface_eager_polars(sample_data1, extension, tmp_path):
    dataif = DataInterface(tmp=tmp_path)

    df = pl.DataFrame(sample_data1)

    dataif.dump(df, "data" + extension, key="tmp")

    loaded_data = dataif.load("data" + extension, key="tmp")

    for key in ["a", "b"]:
        assert np.allclose(sample_data1[key], loaded_data[key])


@pytest.mark.unit
@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_data_interface_lazy_polars(sample_data1, extension, tmp_path):
    dataif = DataInterface(tmp=tmp_path)

    df = pl.LazyFrame(sample_data1)

    dataif.dump(df, "data" + extension, key="tmp")

    lf = dataif.load(
        "data" + extension, key="tmp", return_type="polars_lazyframe"
    )

    assert type(lf) is pl.LazyFrame

    loaded_data = lf.collect()

    for key in ["a", "b"]:
        assert np.allclose(sample_data1[key], loaded_data[key])


@pytest.mark.unit
@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_data_interface_eager_pandas(sample_data1, extension, tmp_path):
    dataif = DataInterface(tmp=tmp_path)

    df = pd.DataFrame(sample_data1)

    dataif.dump(df, "data" + extension, key="tmp")

    loaded_data = dataif.load(
        "data" + extension, key="tmp", return_type="pandas_dataframe"
    )

    for key in ["a", "b"]:
        assert np.allclose(sample_data1[key], loaded_data[key])


@pytest.mark.unit
def test_add_path(tmp_path):
    dataif = DataInterface()

    assert len(dataif.paths) == 0

    dataif.add_path("tmp", tmp_path)

    assert len(dataif.paths) == 1
    assert "tmp" in dataif.paths
    assert dataif.paths["tmp"] == tmp_path


@pytest.mark.unit
def test_add_path_exist_ok(tmp_path):
    dataif = DataInterface(tmp=tmp_path)

    with pytest.raises(ValueError):
        dataif.add_path("tmp", tmp_path)

    dataif.add_path("tmp", tmp_path, exist_ok=True)


@pytest.mark.unit
def test_remove_path(tmp_path):
    dataif = DataInterface(tmp=tmp_path)

    assert len(dataif.paths) == 1

    dataif.remove_path("tmp")

    assert len(dataif.paths) == 0


@pytest.fixture
def data_files(sample_data2, tmp_path):
    """Create small CSV and Parquet files for testing."""
    data = pl.DataFrame(sample_data2)
    csv_path = tmp_path / "data.csv"
    parquet_path = tmp_path / "data.parquet"

    data.write_csv(csv_path)
    data.write_parquet(parquet_path)

    return {"csv": csv_path, "parquet": parquet_path}


@pytest.mark.unit
@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_load_dump_basic(data_files, tmp_path, extension):
    """Test basic load and dump functionality without filtering."""
    dataif = DataInterface(tmp=tmp_path)
    data_path = data_files[extension]

    # Load the data
    loaded_data = dataif.load(data_path.name, key="tmp")

    # Verify that loaded data matches the original sample data
    assert loaded_data.shape == (4, 4)
    assert np.array_equal(loaded_data["age_group_id"], [1, 2, 2, 3])
    assert np.array_equal(loaded_data["location_id"], [10, 20, 20, 30])
    assert np.array_equal(loaded_data["sex_id"], [1, 2, 1, 2])
    assert np.array_equal(loaded_data["value"], [100, 200, 300, 400])


@pytest.mark.unit
@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_load_with_columns(data_files, tmp_path, extension):
    """Test loading with specified columns."""
    dataif = DataInterface(tmp=tmp_path)
    data_path = data_files[extension]

    # Load only specific columns
    loaded_data = dataif.load(
        data_path.name, key="tmp", columns=["age_group_id", "value"]
    )

    # Verify that only the specified columns are loaded
    assert loaded_data.shape == (4, 2)
    assert "age_group_id" in loaded_data.columns
    assert "value" in loaded_data.columns
    assert "location_id" not in loaded_data.columns
    assert "sex_id" not in loaded_data.columns


@pytest.mark.unit
@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_load_with_subset(data_files, tmp_path, extension):
    """Test loading with subset for row filtering."""
    dataif = DataInterface(tmp=tmp_path)
    data_path = data_files[extension]

    subset = {"location_id": 20, "sex_id": [1]}

    # Load with subset filtering
    loaded_data = dataif.load(data_path.name, key="tmp", subset=subset)

    # Verify that only rows matching the subset criteria are loaded
    assert loaded_data.shape == (1, 4)
    assert loaded_data["age_group_id"][0] == 2
    assert loaded_data["location_id"][0] == 20
    assert loaded_data["sex_id"][0] == 1
    assert loaded_data["value"][0] == 300


@pytest.mark.unit
@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_load_with_columns_and_subset(data_files, tmp_path, extension):
    """Test loading with both columns selection and subset filtering."""
    dataif = DataInterface(tmp=tmp_path)
    data_path = data_files[extension]

    columns = ["age_group_id", "location_id", "value"]
    subset = {"location_id": [20]}

    # Load with both columns and subset filters
    loaded_data = dataif.load(
        data_path.name, key="tmp", columns=columns, subset=subset
    )

    # Verify that data is filtered by both columns and subset
    assert loaded_data.shape == (2, 3)
    assert "age_group_id" in loaded_data.columns
    assert "value" in loaded_data.columns
    assert "location_id" in loaded_data.columns
    assert "sex_id" not in loaded_data.columns
    assert np.array_equal(loaded_data["age_group_id"], [2, 2])
    assert np.array_equal(loaded_data["location_id"], [20, 20])
    assert np.array_equal(loaded_data["value"], [200, 300])


@pytest.mark.unit
@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_load_pandas_with_columns_and_subset(data_files, tmp_path, extension):
    """Test loading passes through subset and columns to pandas"""
    dataif = DataInterface(tmp=tmp_path)
    data_path = data_files[extension]

    columns = ["age_group_id", "location_id", "value"]
    subset = {"location_id": [20]}

    with (
        patch(
            "pandas.read_parquet", return_value=pd.DataFrame(columns=columns)
        ) as mock_pd_read_parquet,
        patch(
            "pandas.read_csv", return_value=pd.DataFrame(columns=columns)
        ) as mock_pd_read_csv,
    ):
        dataif.load(data_path.name, key="tmp", columns=columns, subset=subset)
        if extension == "parquet":
            mock_pd_read_parquet.assert_called_once()
            assert "columns" in mock_pd_read_parquet.call_args.kwargs
            assert "filters" in mock_pd_read_parquet.call_args.kwargs
        else:
            mock_pd_read_csv.assert_called_once()
            assert "usecols" in mock_pd_read_csv.call_args.kwargs


@pytest.mark.unit
@pytest.mark.parametrize("fextn", [".json", ".yaml", ".pkl"])
def test_data_interface_with_config_dict(sample_config, fextn, tmp_path):
    configif = DataInterface(config=tmp_path)

    file_name = f"config{fextn}"
    configif.dump(sample_config, file_name, key="config")

    loaded_config = configif.load(file_name, key="config")

    assert loaded_config == sample_config
