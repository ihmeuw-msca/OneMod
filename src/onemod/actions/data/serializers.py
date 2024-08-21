
from pathlib import Path


def path_to_str(p: str | Path) -> str:
    return str(p)


def str_to_path(s: str) -> Path:
    return Path(s)
