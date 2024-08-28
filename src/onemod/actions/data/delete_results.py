"""Delete onemod stage results."""

import shutil
from pathlib import Path

from jobmon.core.task_generator import task_generator

from onemod.actions.data.serializers import str_to_path

@task_generator(
    serializers={Path: (str, str_to_path)},
    tool_name="onemod_tool",
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

