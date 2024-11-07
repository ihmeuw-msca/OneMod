from pathlib import Path

import polars as pl

from onemod.datatools.directory_manager import DirectoryManager
from onemod.datatools.io import FileIO, dataio_dict


class DataInterface(DirectoryManager):
    """Data interface that store important directories and automatically read
    and write data to the stored directories based on their data types.

    Attributes
    ----------
    io_dict : dict[str, FileIO]
        A dictionary that maps the file extensions to the corresponding data io
        class. This is a module-level variable from onemo.datatools.io.dataio_dict.

    """

    io_dict: dict[str, FileIO] = dataio_dict

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
        return self.io_dict[path.suffix].load(path, **options)

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
        self.io_dict[path.suffix].dump(obj, path, **options)

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, supports_lazy_loading=True\n)"
