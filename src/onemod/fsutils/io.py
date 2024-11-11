import json
from abc import ABC
from pathlib import Path
from typing import Any, Type

import dill
import polars as pl
import tomli
import tomli_w
import yaml


class DataIO(ABC):
    """Bridge class that unifies the I/O for different data file types."""

    fextns: tuple[str, ...] = ("",)
    dtypes: tuple[Type, ...] = (pl.DataFrame,)

    def load_eager(self, fpath: str | Path, **options) -> Any:
        fpath = Path(fpath)
        if fpath.suffix not in self.fextns:
            raise ValueError(f"File extension must be in {self.fextns}.")
        return self.load_eager(fpath, **options)

    def load_lazy(self, fpath: str | Path, **options) -> pl.LazyFrame:
        fpath = Path(fpath)
        if fpath.suffix not in self.fextns:
            raise ValueError(f"File extension must be in {self.fextns}.")
        return self.load_lazy(fpath, **options)

    def dump(self, obj: Any, fpath: str | Path, **options):
        fpath = Path(fpath)
        if not isinstance(obj, self.dtypes):
            raise TypeError(f"Data must be an instance of {self.dtypes}.")
        self.dump(obj, fpath, **options)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ConfigIO(ABC):
    """Bridge class that unifies the I/O for different config file types."""

    fextns: tuple[str, ...] = ("",)
    dtypes: tuple[Type, ...] = (object,)

    def load(self, fpath: str | Path, **options) -> Any:
        """Load data from given path."""
        fpath = Path(fpath)
        if fpath.suffix not in self.fextns:
            raise ValueError(f"File extension must be in {self.fextns}.")
        return self.load(fpath, **options)

    def dump(self, obj: Any, fpath: str | Path, mkdir: bool = True, **options):
        """Dump data to given path."""
        fpath = Path(fpath)
        if not isinstance(obj, self.dtypes):
            raise TypeError(f"Data must be an instance of {self.dtypes}.")
        if mkdir:
            fpath.parent.mkdir(parents=True, exist_ok=True)
        self.dump(obj, fpath, **options)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CSVIO(DataIO):
    fextns: tuple[str, ...] = (".csv",)
    dtypes: tuple[Type, ...] = (pl.DataFrame, pl.LazyFrame)

    def load_eager(self, path: Path, **options) -> pl.DataFrame:
        df = pl.read_csv(path, **options)
        return df

    def load_lazy(self, fpath: Path, **options) -> pl.LazyFrame:
        lf = pl.scan_csv(fpath, **options)
        return lf

    def dump(self, obj: pl.DataFrame, path: Path, **options):
        obj.write_csv(path, **options)


class ParquetIO(DataIO):
    fextns: tuple[str, ...] = (".parquet",)
    dtypes: tuple[Type, ...] = (pl.DataFrame, pl.LazyFrame)

    def load_eager(self, fpath: Path, **options) -> pl.DataFrame:
        df = pl.read_parquet(fpath, **options)
        return df

    def load_lazy(self, fpath: Path, **options) -> pl.LazyFrame:
        lf = pl.scan_parquet(fpath, **options)
        return lf

    def dump(self, obj: pl.DataFrame, fpath: Path, **options):
        obj.write_parquet(fpath, **options)


class JSONIO(ConfigIO):
    fextns: tuple[str, ...] = (".json",)
    dtypes: tuple[Type, ...] = (dict, list)

    def load(self, fpath: Path, **options) -> dict | list:
        with open(fpath, "r") as f:
            return json.load(f, **options)

    def dump(self, obj: dict | list, fpath: Path, **options):
        with open(fpath, "w") as f:
            json.dump(obj, f, **options)


class PickleIO(ConfigIO):
    fextns: tuple[str, ...] = (".pkl", ".pickle")

    def load(self, fpath: Path, **options) -> Any:
        with open(fpath, "rb") as f:
            return dill.load(f, **options)

    def dump(self, obj: Any, fpath: Path, **options):
        with open(fpath, "wb") as f:
            return dill.dump(obj, f, **options)


class TOMLIO(ConfigIO):
    fextns: tuple[str, ...] = (".toml",)
    dtypes: tuple[Type, ...] = (dict,)

    def load(self, fpath: Path, **options) -> dict:
        with open(fpath, "rb") as f:
            return tomli.load(f, **options)

    def dump(self, obj: dict, fpath: Path, **options):
        with open(fpath, "wb") as f:
            tomli_w.dump(obj, f)


class YAMLIO(ConfigIO):
    fextns: tuple[str, ...] = (".yml", ".yaml")
    dtypes: tuple[Type, ...] = (dict, list)

    def load(self, fpath: Path, **options) -> dict | list:
        options = dict(Loader=yaml.SafeLoader) | options
        with open(fpath, "r") as f:
            return yaml.load(f, **options)

    def dump(self, obj: dict | list, fpath: Path, **options):
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
