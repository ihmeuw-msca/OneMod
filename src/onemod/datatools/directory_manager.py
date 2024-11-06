from pathlib import Path


class DirectoryManager:
    """Utility for managing directories in DataInterface and ConfigInterface."""

    def __init__(self, **dirs: dict[str, Path | str]) -> None:
        """Initialize directories and set up attributes for each directory."""
        self.dirs = {key: Path(path) for key, path in dirs.items()}
        self.keys = []
        for key, path in dirs.items():
            self.add_dir(key, path)

    def add_dir(
        self, key: str, path: Path | str, exist_ok: bool = False
    ) -> None:
        if not exist_ok and key in self.dirs:
            raise ValueError(f"{key} already exists")
        self.dirs[key] = Path(path)

    def get_dir(self, key: str) -> Path:
        if key not in self.dirs:
            raise ValueError(f"Directory '{key}' not found.")
        return self.dirs[key]

    def remove_dir(self, key: str) -> None:
        if key not in self.dirs:
            raise ValueError(f"Directory '{key}' not found.")
        del self.dirs[key]

    def get_fpath(self, *fparts: tuple[str, ...], key: str = "") -> Path:
        """Retrieve the full path based on key and sub-paths."""
        base_dir = self.get_dir(key)
        return base_dir / "/".join(map(str, fparts))

    def __repr__(self) -> str:
        expr = f"{type(self).__name__}(\n"
        for key in self.keys:
            expr += f"    {key}={getattr(self, key)},\n"
        expr += ")"
        return expr
