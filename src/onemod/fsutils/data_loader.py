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
        backend: Literal[
            "polars", "polars_lazy", "polars_eager", "pandas"
        ] = "polars",
        columns: list[str] | None = None,
        id_subsets: dict[str, list] | None = None,
        **options,
    ) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
        """Load data with optional lazy loading and subset filtering. Supports
        both Polars and Pandas backends."""
        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported data format for '{path.suffix}'")

        if backend in ["polars", "polars_eager"]:
            polars_df = self.io_dict[path.suffix].load_eager(
                path, backend="polars", **options
            )

            if not isinstance(polars_df, pl.DataFrame):
                raise TypeError(
                    f"Expected a Polars DataFrame, got {type(polars_df)}"
                )

            if columns:
                polars_df = polars_df.select(columns)

            if id_subsets:
                for col, values in id_subsets.items():
                    polars_df = polars_df.filter(pl.col(col).is_in(values))

            return polars_df
        elif backend == "polars_lazy":
            polars_lf = self.io_dict[path.suffix].load_lazy(path, **options)

            if columns:
                polars_lf = polars_lf.select(columns)

            if id_subsets:
                for col, values in id_subsets.items():
                    polars_lf = polars_lf.filter(pl.col(col).is_in(values))

            return polars_lf
        elif backend == "pandas":
            pandas_df = self.io_dict[path.suffix].load_eager(
                path, backend="pandas", **options
            )

            if not isinstance(pandas_df, pd.DataFrame):
                raise TypeError(
                    f"Expected a Pandas DataFrame, got {type(pandas_df)}"
                )

            if columns:
                pandas_df = pandas_df[columns]

            if id_subsets:
                for col, values in id_subsets.items():
                    pandas_df = pandas_df[pandas_df[col].isin(values)]

            return pandas_df
        else:
            raise ValueError(
                "Backend must be one of 'polars', 'polars_lazy', 'polars_eager', or 'pandas'"
            )

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
        return f"{type(self).__name__}()"
