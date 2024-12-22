import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Type

import dill
import pandas as pd
import polars as pl
import tomli
import tomli_w
import yaml


class DataIO(ABC):
    """Bridge class that unifies the I/O for different data file types."""

    fextns: tuple[str, ...] = ("",)
    dtypes: tuple[Type, ...] = (pd.DataFrame, pl.DataFrame, pl.LazyFrame)

    def load_eager(
        self,
        fpath: Path | str,
        backend: Literal["pandas", "polars"] = "pandas",
        **options,
    ) -> pd.DataFrame | pl.DataFrame:
        """Load data from given path."""
        fpath = Path(fpath)
        if fpath.suffix not in self.fextns:
            raise ValueError(f"File extension must be in {self.fextns}.")
        if backend == "polars":
            return self._load_eager_polars_impl(fpath, **options)
        elif backend == "pandas":
            return self._load_eager_pandas_impl(fpath, **options)
        else:
            raise ValueError("Backend must be either 'polars' or 'pandas'.")

    def load_lazy(self, fpath: Path | str, **options) -> pl.LazyFrame:
        """Load data lazily (Polars only) from given path."""
        fpath = Path(fpath)
        if fpath.suffix not in self.fextns:
            raise ValueError(f"File extension must be in {self.fextns}.")
        return self._load_lazy_impl(fpath, **options)

    def dump(
        self,
        obj: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        fpath: Path | str,
        mkdir: bool = True,
        **options,
    ):
        """Dump data to given path."""
        fpath = Path(fpath)
        if mkdir:
            fpath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, pl.LazyFrame):
            obj = obj.collect()

        if isinstance(obj, pl.DataFrame):
            self._dump_polars_impl(obj, fpath, **options)
        elif isinstance(obj, pd.DataFrame):
            self._dump_pandas_impl(obj, fpath, **options)
        else:
            raise TypeError(f"Data must be an instance of {self.dtypes}.")

    @abstractmethod
    def _load_eager_polars_impl(self, fpath: Path, **options) -> pl.DataFrame:
        """Polars implementation of eager loading."""
        pass

    @abstractmethod
    def _load_eager_pandas_impl(self, fpath: Path, **options) -> pd.DataFrame:
        """Pandas implementation of eager loading."""
        pass

    @abstractmethod
    def _load_lazy_impl(self, fpath: Path, **options) -> pl.LazyFrame:
        """Polars implementation of lazy loading."""
        pass

    @abstractmethod
    def _dump_polars_impl(self, obj: pl.DataFrame, fpath: Path, **options):
        """Polars implementation of dumping."""
        pass

    @abstractmethod
    def _dump_pandas_impl(self, obj: pd.DataFrame, fpath: Path, **options):
        """Pandas implementation of dumping."""
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ConfigIO(ABC):
    """Bridge class that unifies the I/O for different config file types."""

    fextns: tuple[str, ...] = ("",)
    dtypes: tuple[Type, ...] = (object,)

    def load(self, fpath: Path | str, **options) -> Any:
        """Load data from given path."""
        fpath = Path(fpath)
        if fpath.suffix not in self.fextns:
            raise ValueError(f"File extension must be in {self.fextns}.")
        return self._load_impl(fpath, **options)

    def dump(self, obj: Any, fpath: Path | str, mkdir: bool = True, **options):
        """Dump data to given path."""
        fpath = Path(fpath)
        if not isinstance(obj, self.dtypes):
            raise TypeError(f"Data must be an instance of {self.dtypes}.")
        if mkdir:
            fpath.parent.mkdir(parents=True, exist_ok=True)
        self._dump_impl(obj, fpath, **options)

    @abstractmethod
    def _load_impl(self, fpath: Path, **options) -> Any:
        """Implementation of loading to be provided by subclasses."""
        pass

    @abstractmethod
    def _dump_impl(self, obj: Any, fpath: Path, **options):
        """Implementation of dumping to be provided by subclasses."""
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CSVIO(DataIO):
    fextns: tuple[str, ...] = (".csv",)

    def _load_eager_polars_impl(self, fpath: Path, **options) -> pl.DataFrame:
        return pl.read_csv(fpath, **options)

    def _load_eager_pandas_impl(self, fpath: Path, **options) -> pd.DataFrame:
        return pd.read_csv(fpath, **options)

    def _load_lazy_impl(self, fpath: Path, **options) -> pl.LazyFrame:
        return pl.scan_csv(fpath, **options)

    def _dump_polars_impl(self, obj: pl.DataFrame, fpath: Path, **options):
        obj.write_csv(fpath, **options)

    def _dump_pandas_impl(self, obj: pd.DataFrame, fpath: Path, **options):
        obj.to_csv(fpath, index=False, **options)


class ParquetIO(DataIO):
    fextns: tuple[str, ...] = (".parquet",)

    def _load_eager_polars_impl(self, fpath: Path, **options) -> pl.DataFrame:
        return pl.read_parquet(fpath, **options)

    def _load_eager_pandas_impl(self, fpath: Path, **options) -> pd.DataFrame:
        return pd.read_parquet(fpath, **options)

    def _load_lazy_impl(self, fpath: Path, **options) -> pl.LazyFrame:
        return pl.scan_parquet(fpath, **options)

    def _dump_polars_impl(self, obj: pl.DataFrame, fpath: Path, **options):
        obj.write_parquet(fpath, **options)

    def _dump_pandas_impl(self, obj: pd.DataFrame, fpath: Path, **options):
        obj.to_parquet(fpath, index=False, **options)


class JSONIO(ConfigIO):
    fextns: tuple[str, ...] = (".json",)
    dtypes: tuple[Type, ...] = (dict, list)

    def _load_impl(self, fpath: Path, **options) -> dict | list:
        with open(fpath, "r") as f:
            return json.load(f, **options)

    def _dump_impl(self, obj: dict | list, fpath: Path, **options):
        with open(fpath, "w") as f:
            json.dump(obj, f, **options)


class PickleIO(ConfigIO):
    fextns: tuple[str, ...] = (".pkl", ".pickle")

    def _load_impl(self, fpath: Path, **options) -> Any:
        with open(fpath, "rb") as f:
            return dill.load(f, **options)

    def _dump_impl(self, obj: Any, fpath: Path, **options):
        with open(fpath, "wb") as f:
            return dill.dump(obj, f, **options)


class TOMLIO(ConfigIO):
    fextns: tuple[str, ...] = (".toml",)
    dtypes: tuple[Type, ...] = (dict,)

    def _load_impl(self, fpath: Path, **options) -> dict:
        with open(fpath, "rb") as f:
            return tomli.load(f, **options)

    def _dump_impl(self, obj: dict, fpath: Path, **options):
        with open(fpath, "wb") as f:
            tomli_w.dump(obj, f)


class YAMLIO(ConfigIO):
    fextns: tuple[str, ...] = (".yml", ".yaml")
    dtypes: tuple[Type, ...] = (dict, list)

    def _load_impl(self, fpath: Path, **options) -> dict | list:
        options = dict(Loader=yaml.SafeLoader) | options
        with open(fpath, "r") as f:
            return yaml.load(f, **options)

    def _dump_impl(self, obj: dict | list, fpath: Path, **options):
        options = dict(Dumper=yaml.SafeDumper) | options
        with open(fpath, "w") as f:
            return yaml.dump(obj, f, **options)


dataio_dict: dict[str, DataIO] = {
    fextn: io for io in [CSVIO(), ParquetIO()] for fextn in io.fextns
}

configio_dict: dict[str, ConfigIO] = {
    fextn: io
    for io in [JSONIO(), PickleIO(), TOMLIO(), YAMLIO()]
    for fextn in io.fextns
}
