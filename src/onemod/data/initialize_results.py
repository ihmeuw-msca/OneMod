"""Initialize onemod stage results."""
from pathlib import Path
import shutil
from typing import Union

import fire

from onemod.utils import (
    get_ensemble_input,
    get_rover_submodels,
    get_swimr_submodels,
    get_weave_submodels,
    load_settings,
    Subsets,
)


def initialize_rover_results(experiment_dir: Union[Path, str]) -> None:
    """Initialize rover results."""
    experiment_dir = Path(experiment_dir)
    rover_dir = experiment_dir / "results" / "rover"

    # Initialize directories
    if rover_dir.exists():
        shutil.rmtree(rover_dir)
    for sub_dir in ["data", "submodels"]:
        (rover_dir / sub_dir).mkdir(parents=True)

    # Create rover subsets
    get_rover_submodels(experiment_dir, save_file=True)


def initialize_swimr_results(experiment_dir: Union[Path, str]) -> None:
    """Initialize swimr results."""
    experiment_dir = Path(experiment_dir)
    swimr_dir = experiment_dir / "results" / "swimr"

    # Initialize directories
    if swimr_dir.exists():
        shutil.rmtree(swimr_dir)
    for sub_dir in ["data", "submodels"]:
        (swimr_dir / sub_dir).mkdir(parents=True)

    # Create swimr parameters and subsets
    get_swimr_submodels(experiment_dir, save_files=True)


def initialize_weave_results(experiment_dir: Union[Path, str]) -> None:
    """Initialize weave results."""
    experiment_dir = Path(experiment_dir)
    weave_dir = experiment_dir / "results" / "weave"

    # Initialize directories
    if weave_dir.exists():
        shutil.rmtree(weave_dir)
    (weave_dir / "submodels").mkdir(parents=True)

    # Create weave parameters and subsets
    get_weave_submodels(experiment_dir, save_files=True)


def initialize_ensemble_results(experiment_dir: Union[Path, str]) -> None:
    """Initialize ensemble results."""
    experiment_dir = Path(experiment_dir)
    ensemble_dir = experiment_dir / "results" / "ensemble"

    # Initialize directory
    if ensemble_dir.exists():
        shutil.rmtree(ensemble_dir)
    ensemble_dir.mkdir()

    # Create ensemble subsets
    settings = load_settings(experiment_dir / "config" / "settings.yml")
    if "groupby" in settings["ensemble"]:
        Subsets(
            "ensemble", settings["ensemble"], get_ensemble_input(settings)
        ).subsets.to_csv(ensemble_dir / "subsets.csv", index=False)


def main() -> None:
    fire.Fire(
        {
            "rover": initialize_rover_results,
            "swimr": initialize_swimr_results,
            "weave": initialize_weave_results,
            "ensemble": initialize_ensemble_results,
        }
    )
