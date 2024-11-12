from pathlib import Path

import polars as pl

from onemod.fsutils.io import DataIO, dataio_dict


class DataLoader:
    """Handles loading and dumping of data files with optional lazy loading,
    column selection, and subset filtering."""

    io_dict: dict[str, DataIO] = dataio_dict

    def load(
        self,
        path: Path,
        lazy: bool = False,
        columns: list[str] | None = None,
        id_subsets: dict[str, list] | None = None,
        **options,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Load data with optional lazy loading and subset filtering."""
        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported data format for '{path.suffix}'")

        lf = self.io_dict[path.suffix].load_lazy(path, **options)

        if columns:
            lf = lf.select(columns)

        if id_subsets:
            for col, values in id_subsets.items():
                lf = lf.filter(pl.col(col).is_in(values))

        return lf if lazy else lf.collect()

    def dump(
        self, obj: pl.DataFrame | pl.LazyFrame, path: Path, **options
    ) -> None:
        """Save data file."""
        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported data format for '{path.suffix}'")

        self.io_dict[path.suffix].dump(obj, path, **options)

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, supports_lazy_loading=True\n)"
