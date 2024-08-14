"""Delete onemod stage results."""

import os
import shutil
from pathlib import Path

import fire

from jobmon.core.task_generator import task_generator

script_path = os.path.abspath(__file__)
# Resolve any symbolic links (if necessary)
full_script_path = os.path.realpath(script_path)


@task_generator(
    serializers={},
    tool_name="onemod_tool",
    module_source_path=full_script_path,
    max_attempts=2,
    naming_args=["result"],
)
def delete_result(result: str | Path) -> None:
    """Delete result directory or file."""
    result = Path(result)
    if result.is_dir():
        shutil.rmtree(result)
    else:
        result.unlink(missing_ok=True)


def main() -> None:
    fire.Fire(delete_result)
