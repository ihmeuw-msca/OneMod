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

        if return_type == "pandas_dataframe":
            pandas_df = self.io_dict[path.suffix].load_eager(path, **options)
            assert isinstance(
                pandas_df, pd.DataFrame
            ), "Expected a pandas DataFrame"

            if columns:
                pandas_df = pandas_df[columns]

            if subset:
                for col, values in subset.items():
                    pandas_df = pandas_df[
                        pandas_df[col].isin(
                            values if isinstance(values, list) else [values]
                        )
                    ]
                    pandas_df.reset_index(drop=True, inplace=True)

            return pandas_df
        elif return_type in ["polars_dataframe", "polars_lazyframe"]:
            polars_lf = self.io_dict[path.suffix].load_lazy(path, **options)

            if columns:
                polars_lf = polars_lf.select(columns)

            if subset:
                for col, values in subset.items():
                    polars_lf = polars_lf.filter(
                        pl.col(col).is_in(
                            values if isinstance(values, list) else [values]
                        )
                    )

            if return_type == "polars_dataframe":
                return polars_lf.collect()
            elif return_type == "polars_lazyframe":
                return polars_lf
        else:
            raise ValueError(
                "Return type must be one of 'polars_dataframe', 'polars_lazyframe', or 'pandas_dataframe'"
            )

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
