from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd
import polars as pl

from onemod.fsutils.config_loader import ConfigLoader
from onemod.fsutils.data_loader import DataLoader
from onemod.fsutils.path_manager import PathManager


class DataInterface(PathManager):
    """Unified interface for loading and dumping data and config files."""

    def __init__(self, **paths: Path | str) -> None:
        super().__init__(**paths)
        self.data_loader = DataLoader()
        self.config_loader = ConfigLoader()

    def load(
        self,
        *fparts: str,
        key: str | None = None,
        return_type: Literal[
            "pandas_dataframe", "polars_dataframe", "polars_lazyframe"
        ] = "pandas_dataframe",
        columns: list[str] | None = None,
        subset: Mapping[str, Any | list[Any]] | None = None,
        **options,
    ) -> Any:
        """Load data or config files based on file type and key.

        Parameters
        ----------
        return_type : {'pandas_dataframe', 'polars_dataframe', 'polars_lazyframe'}, optional
            Return type of loaded data object, applicable only for data files.
        columns : list of str, optional
            Specific columns to load, applicable only for data files.
        subset : dict, optional
            Subset of column values for filtering, applicable only for
            data files.
        options : dict
            Additional options for loading.

        Returns
        -------
        Any
            Loaded data or config file.
        """
        path = self.get_full_path(*fparts, key=key)
        if path.suffix in self.config_loader.io_dict.keys():
            return self.config_loader.load(path, **options)
        elif path.suffix in self.data_loader.io_dict.keys():
            return self.data_loader.load(
                path,
                return_type=return_type,
                columns=columns,
                subset=subset,
                **options,
            )
        else:
            raise ValueError(f"Unsupported file format for '{path.suffix}'")

    def dump(
        self, obj: Any, *fparts: str, key: str | None = None, **options
    ) -> None:
        """Dump data or config files based on object type and key."""
        path = self.get_full_path(*fparts, key=key)
        if isinstance(obj, (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
            return self.data_loader.dump(obj, path, **options)
        else:
            return self.config_loader.dump(obj, path, **options)
