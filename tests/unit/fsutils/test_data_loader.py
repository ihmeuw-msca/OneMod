from typing import Any, Mapping
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from onemod.fsutils.data_loader import DataLoader


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["x", "y", "x", "z"], "c": [9, 8, 7, 6]}
    )


@pytest.mark.unit
@pytest.mark.parametrize("extension", [".csv", ".parquet"])
@pytest.mark.parametrize(
    "return_type, columns, subset, expected_fn",
    [
        # Pandas
        ("pandas_dataframe", None, None, lambda df: df),
        ("pandas_dataframe", ["b", "a"], None, lambda df: df[["b", "a"]]),
        (
            "pandas_dataframe",
            None,
            {"b": ["x"]},
            lambda df: df[df["b"].isin(["x"])].reset_index(drop=True),
        ),
        (
            "pandas_dataframe",
            ["a", "c"],
            {"a": [2, 4]},
            lambda df: df[df["a"].isin([2, 4])][["a", "c"]].reset_index(
                drop=True
            ),
        ),
        # Polars
        ("polars_dataframe", None, None, lambda df: pl.DataFrame(df)),
        (
            "polars_dataframe",
            ["c", "a"],
            None,
            lambda df: pl.DataFrame(df[["c", "a"]]),
        ),
        (
            "polars_dataframe",
            None,
            {"b": ["y", "z"]},
            lambda df: pl.DataFrame(
                df[df["b"].isin(["y", "z"])].reset_index(drop=True)
            ),
        ),
        (
            "polars_dataframe",
            ["b", "c"],
            {"a": [1, 3]},
            lambda df: pl.DataFrame(
                df[df["a"].isin([1, 3])][["b", "c"]].reset_index(drop=True)
            ),
        ),
        # Polars Lazy
        ("polars_lazyframe", None, None, lambda df: pl.DataFrame(df)),
        (
            "polars_lazyframe",
            ["a"],
            {"b": ["z"]},
            lambda df: pl.DataFrame(
                df[df["b"] == "z"][["a"]].reset_index(drop=True)
            ),
        ),
    ],
)
def test_load(
    tmp_path: str,
    extension: str,
    data: pd.DataFrame,
    return_type: str,
    columns: list[str] | None,
    subset: Mapping[str, Any | list[Any]] | None,
    expected_fn,
) -> None:
    """Test data can be loaded via parquet and CSV."""
    if extension == ".csv":
        path = tmp_path / "test.csv"
        data.to_csv(path, index=False)
    elif extension == ".parquet":
        path = tmp_path / "test.parquet"
        data.to_parquet(path, index=False)

    loader = DataLoader()
    out = loader.load(
        path, return_type=return_type, columns=columns, subset=subset
    )
    expected = expected_fn(data)
    if return_type == "pandas_dataframe":
        pd.testing.assert_frame_equal(out, expected)
    elif return_type == "polars_dataframe":
        assert out.equals(expected)
    elif return_type == "polars_lazyframe":
        collected = out.collect()
        assert collected.equals(expected)
    else:
        raise AssertionError("Unexpected return_type")


@pytest.mark.unit
def test_csv_columns_usecols_warning(tmp_path: str, data: pd.DataFrame) -> None:
    """Test that warning is raised when both columns and use_cols are passed."""
    path = tmp_path / "test.csv"
    data.to_csv(path, index=False)
    loader = DataLoader()
    with pytest.warns(UserWarning, match="Both `columns` and `usecols` passed"):
        out = loader.load(
            path,
            return_type="pandas_dataframe",
            columns=["a", "b"],
            usecols=["a", "b", "c"],
        )
    pd.testing.assert_frame_equal(out, data[["a", "b"]])


@pytest.mark.unit
def test_parquet_subset_filters_warning(
    tmp_path: str, data: pd.DataFrame
) -> None:
    """Test that warning is raised when both subset and filters are passed."""
    path = tmp_path / "test.parquet"
    data.to_parquet(path, index=False)
    loader = DataLoader()
    with pytest.warns(UserWarning, match="Both `subset` and `filters` passed"):
        out = loader.load(
            path,
            return_type="pandas_dataframe",
            subset={"b": ["x", "y"]},
            filters=[("c", "=", 8)],
        )
    expected = data[data["b"].isin(["x", "y"])]
    expected = expected[expected["c"] == 8].reset_index(drop=True)
    pd.testing.assert_frame_equal(out, expected)


@pytest.mark.unit
@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_load_pandas_with_columns_and_subset(
    tmp_path: str, extension: str, data: pd.DataFrame
) -> None:
    """Test loading passes through subset and columns to pandas"""
    if extension == ".csv":
        path = tmp_path / "test.csv"
        data.to_csv(path, index=False)
    elif extension == ".parquet":
        path = tmp_path / "test.parquet"
        data.to_parquet(path, index=False)
    loader = DataLoader()
    columns = ["age_group_id", "location_id", "value"]

    with (
        patch(
            "pandas.read_parquet", return_value=pd.DataFrame(columns=columns)
        ) as mock_pd_read_parquet,
        patch(
            "pandas.read_csv", return_value=pd.DataFrame(columns=columns)
        ) as mock_pd_read_csv,
    ):
        loader.load(path, columns=columns, subset={"location_id": [20]})
        if extension == ".parquet":
            mock_pd_read_parquet.assert_called_once()
            assert "columns" in mock_pd_read_parquet.call_args.kwargs
            assert "filters" in mock_pd_read_parquet.call_args.kwargs
        else:
            mock_pd_read_csv.assert_called_once()
            assert "usecols" in mock_pd_read_csv.call_args.kwargs
