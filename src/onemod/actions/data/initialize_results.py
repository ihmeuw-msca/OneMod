"""Initialize onemod stage results."""

import shutil

import fire
from pplkit.data.interface import DataInterface

from onemod.utils import get_handle, get_submodels


def initialize_results(directory: str, stages: list[str]) -> None:
    stage_init_map: dict[str, callable] = {
        "rover_covsel": _initialize_rover_covsel_results,
        "spxmod": _initialize_spxmod_results,
        "weave": _initialize_weave_results,
        "ensemble": _initialize_ensemble_results,
    }
    dataif, _ = get_handle(directory)
    for stage in stages:
        stage_init_map[stage](dataif)


def _initialize_rover_covsel_results(dataif: DataInterface) -> None:
    """Initialize rover covariate selection results."""

    # Initialize directories
    if dataif.rover_covsel.exists():
        shutil.rmtree(dataif.rover_covsel)
    for sub_dir in ["data", "submodels"]:
        (dataif.rover_covsel / sub_dir).mkdir(parents=True)

    # Create rover subsets
    get_submodels("rover_covsel", dataif.experiment, save_file=True)


def _initialize_spxmod_results(dataif: DataInterface) -> None:
    # Initialize directories
    if dataif.spxmod.exists():
        shutil.rmtree(dataif.spxmod)
    dataif.spxmod.mkdir(parents=True)


def _initialize_weave_results(dataif: DataInterface) -> None:
    """Initialize weave results."""

    # Initialize directories
    if dataif.weave.exists():
        shutil.rmtree(dataif.weave)
    (dataif.weave / "submodels").mkdir(parents=True)

    # Create weave parameters and subsets
    get_submodels("weave", dataif.experiment, save_file=True)


def _initialize_ensemble_results(dataif: DataInterface) -> None:
    """Initialize ensemble results."""

    # Initialize directory
    if dataif.ensemble.exists():
        shutil.rmtree(dataif.ensemble)
    dataif.ensemble.mkdir(parents=True)

    # Create ensemble subsets
    get_submodels("ensemble", dataif.experiment, save_file=True)


def main() -> None:
    fire.Fire(initialize_results)
