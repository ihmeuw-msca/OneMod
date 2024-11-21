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
        return_type: Literal[
            "polars_dataframe", "polars_lazyframe", "pandas_dataframe"
        ] = "polars_dataframe",
        columns: list[str] | None = None,
        id_subsets: dict[str, list] | None = None,
        **options,
    ) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
        """Load data with lazy loading and subset filtering. Polars and
        Pandas options available for the type of the returned data object."""

        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported data format for '{path.suffix}'")

        polars_lf = self.io_dict[path.suffix].load_lazy(path, **options)

        if columns:
            polars_lf = polars_lf.select(columns)

        if id_subsets:
            for col, values in id_subsets.items():
                polars_lf = polars_lf.filter(pl.col(col).is_in(values))

        if return_type == "polars_dataframe":
            return polars_lf.collect()
        elif return_type == "polars_lazyframe":
            return polars_lf
        elif return_type == "pandas_dataframe":
            return polars_lf.collect().to_pandas()
        else:
            raise ValueError(
                "Return type must be one of 'polars_dataframe', 'polars_lazyframe', or 'pandas_dataframe'"
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
