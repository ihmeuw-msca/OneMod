"""Delete onemod stage results."""

import os
import shutil
from pathlib import Path

import fire

from jobmon.core.task_generator import task_generator

script_path = os.path.abspath(__file__)
# Resolve any symbolic links (if necessary)
full_script_path = os.path.realpath(script_path)


def path_to_str(p: Path) -> str:
    return str(p)


def str_to_path(s: str) -> Path:
    return Path(s)


@task_generator(
    serializers={},
    tool_name="onemod_tool",
    module_source_path=full_script_path,
    serialiesers={Path: (str, path_to_str), str: (Path, str_to_path)},
    max_attempts=2,
    naming_args=["result"],
)
def delete_result(result: str | Path) -> None:
    """
    Delete result directory or file.
    Notice the custom serializer for Path.
    It could just be serializers={  Path: (str, Path)  }  but I wanted to show how to use custom serializers.
    """
    result = Path(result)
    if result.is_dir():
        shutil.rmtree(result)
    else:
        result.unlink(missing_ok=True)


def main() -> None:
    fire.Fire(delete_result)
