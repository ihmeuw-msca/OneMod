"""Delete onemod stage results."""

import os
import shutil
from pathlib import Path

from jobmon.core.task_generator import task_generator

from onemod.actions.data.serializers import path_to_str, str_to_path

script_path = os.path.abspath(__file__)
# Resolve any symbolic links (if necessary)
full_script_path = os.path.realpath(script_path)

#  serializers={str | Path: (str, path_to_str), str | Path: (Path, str_to_path)},

@task_generator(
    serializers={Path: (str, str_to_path)},
    tool_name="onemod_tool",
    module_source_path=full_script_path,
    max_attempts=2,
    naming_args=["result"],
)
def delete_results(result: Path ) -> None:
    """
    Delete result directory or file.
    Notice the custom serializer for Path.
    It could just be serializers={  Path: (str, Path)  }  but I wanted to show how to use custom serializers.
    """
    # result = Path(result)
    if result.is_dir():
        shutil.rmtree(result)
    else:
        result.unlink(missing_ok=True)

