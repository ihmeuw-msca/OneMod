from pathlib import Path
from typing import Any, Literal, Mapping

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
            "pandas_dataframe", "polars_dataframe", "polars_lazyframe"
        ] = "pandas_dataframe",
        columns: list[str] | None = None,
        subset: Mapping[str, Any | list[Any]] | None = None,
        **options,
    ) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
        """Load data with lazy loading and subset filtering. Polars and
        Pandas options available for the type of the returned data object."""

        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported data format for '{path.suffix}'")

        if return_type in ["pandas_dataframe", "polars_dataframe"]:
            backend = return_type.split("_")[0]
            obj = self.io_dict[path.suffix].load_eager(
                path, backend=backend, columns=columns, subset=subset, **options
            )
        elif return_type == "polars_lazyframe":
            obj = self.io_dict[path.suffix].load_lazy(
                path, columns=columns, subset=subset, **options
            )
        else:
            raise ValueError(
                "Return type must be one of 'polars_dataframe', 'polars_lazyframe', or 'pandas_dataframe'"
            )

        expected_data_type = {
            "pandas_dataframe": pd.DataFrame,
            "polars_dataframe": pl.DataFrame,
            "polars_lazyframe": pl.LazyFrame,
        }[return_type]
        if not isinstance(obj, expected_data_type):
            raise RuntimeError(
                f"Given return_type='{return_type}', expected loaded data to be "
                f"of type {expected_data_type} not {type(obj)}."
            )

        return obj

    def dump(
        self,
        obj: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        path: Path,
        **options,
    ) -> None:
        """Save data file."""
        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported data format for '{path.suffix}'")

        self.io_dict[path.suffix].dump(obj, path, **options)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
