"""Delete onemod stage results."""

import shutil
from pathlib import Path

import fire


def delete_result(result: str | Path) -> None:
    """Delete result directory or file."""
    result = Path(result)
    if result.is_dir():
        shutil.rmtree(result)
    else:
        result.unlink(missing_ok=True)


def main() -> None:
    fire.Fire(delete_result)
