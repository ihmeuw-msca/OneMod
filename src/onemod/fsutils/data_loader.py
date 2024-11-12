from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

from onemod.fsutils.io import DataIO, dataio_dict


class DataLoader:
    """Handles loading and dumping of data files with optional lazy loading,
    column selection, and subset filtering."""

    io_dict: dict[str, DataIO] = dataio_dict

    def load(
        self,
        path: Path,
        lazy: bool = False,
        backend: Literal["polars", "pandas"] = "polars",
        columns: list[str] | None = None,
        id_subsets: dict[str, list] | None = None,
        **options,
    ) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
        """Load data with optional lazy loading and subset filtering. Supports
        both Polars and Pandas backends."""
        if lazy and backend == "pandas":
            raise ValueError("Pandas backend does not support lazy loading")

        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported data format for '{path.suffix}'")

        if backend == "polars":
            lf = self.io_dict[path.suffix].load_lazy(path, **options)

            if columns:
                lf = lf.select(columns)

            if id_subsets:
                for col, values in id_subsets.items():
                    lf = lf.filter(pl.col(col).is_in(values))

            return lf if lazy else lf.collect()
        elif backend == "pandas":
            df = self.io_dict[path.suffix].load_eager(
                path, backend="pandas", **options
            )

            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected a Pandas DataFrame, got {type(df)}")

            if columns:
                df = df[columns]

            if id_subsets:
                for col, values in id_subsets.items():
                    df = df[df[col].isin(values)]

            return df
        else:
            raise ValueError("Backend must be either 'polars' or 'pandas'")

    def dump(
        self,
        obj: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        path: Path,
        **options,
    ) -> None:
        """Save data file."""
        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported data format for '{path.suffix}'")

        self.io_dict[path.suffix].dump(obj, path, **options)

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, supports_lazy_loading=True\n)"
