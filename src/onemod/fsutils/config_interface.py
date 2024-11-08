from typing import Any

from onemod.fsutils.directory_manager import PathManager
from onemod.fsutils.io import FileIO, configio_dict


class ConfigInterface(PathManager):
    """Interface for handling configuration files and serialized model files.

    Attributes
    ----------
    io_dict : dict[str, FileIO]
        A dictionary that maps the file extensions to the corresponding config io
        class. This is a module-level variable from onemod.fsutils.io.configio_dict.
    """

    io_dict: dict[str, FileIO] = configio_dict

    def load(self, *fparts: str, key: str | None = None, **options) -> Any:
        """Load a config file or serialized model, depending on file extension."""
        path = self.get_full_path(*fparts, key=key)

        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported config format for '{path.suffix}'")

        return self.io_dict[path.suffix].load(path, **options)

    def dump(
        self, obj: Any, *fparts: str, key: str | None = None, **options
    ) -> None:
        """Save a config or model object to a specified path, ensuring the format matches."""
        path = self.get_full_path(*fparts, key=key)

        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported config format for '{path.suffix}'")

        self.io_dict[path.suffix].dump(obj, path, **options)

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, handles_configs=True\n)"
