from typing import Any

from onemod.datatools.directory_manager import DirectoryManager
from onemod.datatools.io import dataio_dict


class ConfigInterface(DirectoryManager):
    """Interface for handling configuration files and serialized model files."""

    def load(self, *fparts: str, key: str = "", **options) -> Any:
        """Load a config file or serialized model, depending on file extension."""
        path = self.get_fpath(*fparts, key=key)
        if path.suffix not in dataio_dict:
            raise ValueError(f"Unsupported config format for '{path.suffix}'")
        return dataio_dict[path.suffix].load(path, **options)

    def dump(self, obj: Any, *fparts: str, key: str = "", **options) -> None:
        """Save a config or model object to a specified path, ensuring the format matches."""
        path = self.get_fpath(*fparts, key=key)
        if path.suffix not in dataio_dict:
            raise ValueError(f"Unsupported config format for '{path.suffix}'")
        dataio_dict[path.suffix].dump(obj, path, **options)

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, handles_configs=True\n)"
