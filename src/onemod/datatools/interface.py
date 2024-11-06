from functools import partial
from pathlib import Path
from typing import Any

from onemod.datatools.io import DataIO, dataio_dict


class DataInterface:
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

    def __init__(self, **dirs: dict[str, str | Path]) -> None:
        self.keys = []
        for key, value in dirs.items():
            self.add_dir(key, value)

    def add_dir(
        self, key: str, value: str | Path, exist_ok: bool = False
    ) -> None:
        """Add a directory to instance. If the directory already exist

        Parameters
        ----------
        key
            Directory name.
        value
            Directory path.
        exist_ok
            If ``exist_ok=True`` and ``key`` already exists in the current
            instance it will raise an error. Otherwise it will overwrite the
            path corresponding to the ``key``.

        Raises
        ------
        ValueError
            Raised when ``exist_ok=False`` and ``key`` already exists.

        """
        if (not exist_ok) and (key in self.keys):
            raise ValueError(f"{key} already exists")
        setattr(self, key, Path(value))
        setattr(self, f"load_{key}", partial(self.load, key=key))
        setattr(self, f"dump_{key}", partial(self.dump, key=key))
        if key not in self.keys:
            self.keys.append(key)

    def remove_dir(self, key: str) -> None:
        """Remove a directory from the current set of directories.

        Parameters
        ----------
        key
            Directory name

        """
        if key in self.keys:
            delattr(self, key)
            delattr(self, f"load_{key}")
            delattr(self, f"dump_{key}")
            self.keys.remove(key)

    def get_fpath(self, *fparts: tuple[str, ...], key: str = "") -> Path:
        """Get the file path from the name of the directory and the sub-parts
        under the directory.

        Parameters
        ----------
        fparts
            Subdirectories or the file name.
        key
            The name of the directory stored in the class.

        """
        return getattr(self, key, Path(".")) / "/".join(map(str, fparts))

    def load(
        self, *fparts: tuple[str, ...], key: str = "", **options: dict[str, Any]
    ) -> Any:
        """Load data from given directory.

        Parameters
        ----------
        fparts
            Subdirectories or the file name.
        key
            The name of the directory stored in the class.
        options
            Extra arguments for the load function.

        Returns
        -------
        Any
            Data loaded from the given path.

        """
        fpath = self.get_fpath(*fparts, key=key)
        return self.dataio_dict[fpath.suffix].load(fpath, **options)

    def dump(
        self,
        obj: Any,
        *fparts: str,
        key: str = "",
        mkdir: bool = True,
        **options: dict[str, Any],
    ):
        """Dump data to the given directory.

        Parameters
        ----------
        obj
            Provided data object.
        fparts
            Subdirectories or the file name.
        key
            The name of the directory stored in the class.
        mkdir
            If true, it will automatically create the parent directory. The
            default is true.
        options
            Extra arguments for the dump function.

        """
        fpath = self.get_fpath(*fparts, key=key)
        self.dataio_dict[fpath.suffix].dump(obj, fpath, mkdir=mkdir, **options)

    def __repr__(self) -> str:
        expr = f"{type(self).__name__}(\n"
        for key in self.keys:
            expr += f"    {key}={getattr(self, key)},\n"
        expr += ")"
        return expr
