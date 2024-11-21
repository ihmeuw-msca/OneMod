from pathlib import Path
from typing import Any

from onemod.fsutils.io import ConfigIO, configio_dict


class ConfigLoader:
    """Handles loading and dumping of configuration files and serialized models."""

    io_dict: dict[str, ConfigIO] = configio_dict

    def load(self, path: Path, **options) -> Any:
        """Load a config file or serialized model."""
        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported config format for '{path.suffix}'")

        return self.io_dict[path.suffix].load(path, **options)

    def dump(self, obj: Any, path: Path, **options) -> None:
        """Save a config or model object."""
        if path.suffix not in self.io_dict:
            raise ValueError(f"Unsupported config format for '{path.suffix}'")

        self.io_dict[path.suffix].dump(obj, path, **options)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
