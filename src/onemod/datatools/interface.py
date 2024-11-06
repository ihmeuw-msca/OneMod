from pathlib import Path

import polars as pl

from onemod.datatools.directory_manager import DirectoryManager
from onemod.datatools.io import DataIO, dataio_dict


class DataInterface(DirectoryManager):
    """Data interface that store important directories and automatically read
    and write data to the stored directories based on their data types.

    Parameters
    ----------
    dirs
        Directories to manage with directory's name as the name of the keyword
        argument's name and directory's path as the value of the keyword
        argument's value.

    """

    dataio_dict: dict[str, DataIO] = dataio_dict
    """A dictionary that maps the file extensions to the corresponding data io
    class. This is a module-level variable from
    :py:data:`pplkit.data.io.dataio_dict`.

    :meta hide-value:

    """

    def __init__(self, **dirs: dict[str, Path | str]) -> None:
        self.dirs = {key: Path(path) for key, path in dirs.items()}
        self.keys = []
        for key, value in dirs.items():
            self.add_dir(key, value)

    def load(
        self,
        *fparts: str,
        key: str = "",
        lazy: bool = False,
        columns: list[str] | None = None,
        id_subsets: dict[str, list] | None = None,
        **options,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Load data with optional lazy loading and subset filtering."""
        path = self.get_fpath(*fparts, key=key)

        if lazy:
            return self._lazy_load(path, columns, id_subsets)
        return self.dataio_dict[path.suffix].load(path, **options)

    def _lazy_load(
        self,
        path: Path,
        columns: list[str] | None = None,
        id_subsets: dict[str, list] | None = None,
    ) -> pl.LazyFrame:
        """Internal lazy load for data files with subset filtering."""
        lf = (
            pl.scan_parquet(path)
            if path.suffix == ".parquet"
            else pl.scan_csv(path)
        )
        lf = lf.select(columns) if columns else lf

        if id_subsets:
            for col, values in id_subsets.items():
                lf = lf.filter(pl.col(col).is_in(values))

        return lf

    def dump(
        self, obj: pl.DataFrame, *fparts: str, key: str = "", **options
    ) -> None:
        """Save data based on directory key and filepath parts."""
        path = self.get_fpath(*fparts, key=key)
        self.dataio_dict[path.suffix].dump(obj, path, **options)

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, supports_lazy_loading=True\n)"
