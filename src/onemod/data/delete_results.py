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


def delete_stage_results(stage_name: str, experiment_dir: Union[Path, str]) -> None:
    """Delete intermediate stage results."""
    experiment_dir = Path(experiment_dir)
    stage_dir = experiment_dir / "results" / stage_name
    for result in stage_dir.iterdir():
        if result.is_dir():
            delete_result(result)


def main() -> None:
    fire.Fire(
        {
            "result": delete_result,
            "stage": delete_stage_results,
        }
    )
