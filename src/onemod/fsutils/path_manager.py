from pathlib import Path


class PathManager:
    """Utility for managing filesystem paths via DataInterface and ConfigInterface."""

    def __init__(self, **paths: Path | str) -> None:
        """Initialize key-value pairs for paths."""
        self.paths: dict[str, Path] = {}
        for key, path in paths.items():
            self.add_path(key, Path(path))

    def add_path(
        self, key: str, path: Path | str, exist_ok: bool = False
    ) -> None:
        if not exist_ok and key in self.paths:
            raise ValueError(f"{key} already exists")
        self.paths[key] = Path(path)

    def get_path(self, key: str) -> Path:
        if key not in self.paths:
            raise ValueError(f"Path for '{key}' not found.")
        return self.paths[key]

    def remove_path(self, key: str) -> None:
        if key not in self.paths:
            raise ValueError(f"Path for '{key}' not found.")
        del self.paths[key]

    def get_full_path(self, *fparts: str, key: str | None = None) -> Path:
        """Retrieve the full path based on key and sub-paths."""
        base_dir = Path("") if key is None else self.get_path(key)
        return base_dir / "/".join(map(str, fparts))

    def __repr__(self) -> str:
        expr = f"{type(self).__name__}(\n"
        for key, path in self.paths.items():
            expr += f"    {key}={path},\n"
        expr += ")"
        return expr
