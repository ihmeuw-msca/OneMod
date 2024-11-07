import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type

import dill
import polars as pl
import tomli
import tomli_w
import yaml


class FileIO(ABC):
    """Bridge class that unifies the I/O for different file types.

    Attributes
    ----------
    fextns : tuple[str, ...]
        The file extensions. When loading a file, it will be used to check if
        the file extension matches.
    dtypes : tuple[Type, ...]
        The data types. When dumping the data, it will be used to check if the
        data type matches.
    """

    fextns: tuple[str, ...] = ("",)
    dtypes: tuple[Type, ...] = (object,)

    @abstractmethod
    def _load(self, fpath: Path, **options) -> Any:
        pass

    @abstractmethod
    def _dump(self, obj: Any, fpath: Path, **options):
        pass

    def load(self, fpath: str | Path, **options) -> Any:
        """Load data from given path.

        Parameters
        ----------
        fpath
            Provided file path.
        options
            Extra arguments for the load function.

        Raises
        ------
        ValueError
            Raised when the file extension doesn't match.

        Returns
        -------
        Any
            Data loaded from the given path.

        """
        fpath = Path(fpath)
        if fpath.suffix not in self.fextns:
            raise ValueError(f"File extension must be in {self.fextns}.")
        return self._load(fpath, **options)

    def dump(self, obj: Any, fpath: str | Path, mkdir: bool = True, **options):
        """Dump data to given path.

        Parameters
        ----------
        obj
            Provided data object.
        fpath
            Provided file path.
        mkdir
            If true, it will automatically create the parent directory. The
            default is true.
        options
            Extra arguments for the dump function.

        Raises
        ------
        TypeError
            Raised when the given data object type doesn't match.

        """
        fpath = Path(fpath)
        if not isinstance(obj, self.dtypes):
            raise TypeError(f"Data must be an instance of {self.dtypes}.")
        if mkdir:
            fpath.parent.mkdir(parents=True, exist_ok=True)
        self._dump(obj, fpath, **options)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CSVIO(FileIO):
    fextns: tuple[str, ...] = (".csv",)
    dtypes: tuple[Type, ...] = (pl.DataFrame,)

    def _load(self, fpath: Path, **options) -> pl.DataFrame:
        return pl.read_csv(fpath, **options)

    def _dump(self, obj: pl.DataFrame, fpath: Path, **options):
        options = dict(index=False) | options
        obj.to_csv(fpath, **options)


class ParquetIO(FileIO):
    fextns: tuple[str, ...] = (".parquet",)
    dtypes: tuple[Type, ...] = (pl.DataFrame,)

    def _load(self, fpath: Path, **options) -> pl.DataFrame:
        options = dict(engine="pyarrow") | options
        return pl.read_parquet(fpath, **options)

    def _dump(self, obj: pl.DataFrame, fpath: Path, **options):
        options = dict(engine="pyarrow") | options
        obj.to_parquet(fpath, **options)


class JSONIO(FileIO):
    fextns: tuple[str, ...] = (".json",)
    dtypes: tuple[Type, ...] = (dict, list)

    def _load(self, fpath: Path, **options) -> dict | list:
        with open(fpath, "r") as f:
            return json.load(f, **options)

    def _dump(self, obj: dict | list, fpath: Path, **options):
        with open(fpath, "w") as f:
            json.dump(obj, f, **options)


class PickleIO(FileIO):
    fextns: tuple[str, ...] = (".pkl", ".pickle")

    def _load(self, fpath: Path, **options) -> Any:
        with open(fpath, "rb") as f:
            return dill.load(f, **options)

    def _dump(self, obj: Any, fpath: Path, **options):
        with open(fpath, "wb") as f:
            return dill.dump(obj, f, **options)


class TOMLIO(FileIO):
    fextns: tuple[str, ...] = (".toml",)
    dtypes: tuple[Type, ...] = (dict,)

    def _load(self, fpath: Path, **options) -> dict:
        with open(fpath, "rb") as f:
            return tomli.load(f, **options)

    def _dump(self, obj: dict, fpath: Path, **options):
        with open(fpath, "wb") as f:
            tomli_w.dump(obj, f)


class YAMLIO(FileIO):
    fextns: tuple[str, ...] = (".yml", ".yaml")
    dtypes: tuple[Type, ...] = (dict, list)

    def _load(self, fpath: Path, **options) -> dict | list:
        options = dict(Loader=yaml.SafeLoader) | options
        with open(fpath, "r") as f:
            return yaml.load(f, **options)

    def _dump(self, obj: dict | list, fpath: Path, **options):
        options = dict(Dumper=yaml.SafeDumper) | options
        with open(fpath, "w") as f:
            return yaml.dump(obj, f, **options)


dataio_dict: dict[str, FileIO] = {
    fextn: io for io in [CSVIO(), ParquetIO()] for fextn in io.fextns
}

configio_dict: dict[str, FileIO] = {
    fextn: io
    for io in [JSONIO(), PickleIO(), TOMLIO(), YAMLIO()]
    for fextn in io.fextns
}
