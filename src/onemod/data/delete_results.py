"""Delete onemod stage results."""
from pathlib import Path
import shutil
from typing import Union

import fire


def delete_result(result: Union[Path, str]) -> None:
    """Delete result directory or file."""
    result = Path(result)
    if result.is_dir():
        shutil.rmtree(result)
    else:
        result.unlink(missing_ok=True)


def main() -> None:
    fire.Fire(delete_result)
