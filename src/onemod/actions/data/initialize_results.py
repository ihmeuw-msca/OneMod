"""Initialize onemod stage results."""
import shutil

import fire
from pplkit.data.interface import DataInterface

from onemod.utils import (
    get_ensemble_input,
    get_handle,
    get_rover_covsel_submodels,
    get_swimr_submodels,
    get_weave_submodels,
    Subsets,
)


def initialize_results(experiment_dir: str, stages: list[str]) -> None:
    stage_init_map: dict[str, callable] = {
        "rover_covsel": _initialize_rover_covsel_results,
        "regmod_smooth": _initialize_regmod_smooth_results,
        "swimr": _initialize_swimr_results,
        "weave": _initialize_weave_results,
        "ensemble": _initialize_ensemble_results,
    }

    dataif, settings = get_handle(experiment_dir)

    for stage in stages:
        stage_init_map[stage](dataif)

    # ETL the input data into parquet format.
    # More compressible, faster IO, allows for partitioning
    raw_input_path = settings.input_path
    data = dataif.load(raw_input_path)

    # Saves to $experiment_dir/data/data.parquet
    dataif.dump_data(data)


def _initialize_rover_covsel_results(dataif: DataInterface) -> None:
    """Initialize rover results."""

    # Initialize directories
    if dataif.rover_covsel.exists():
        shutil.rmtree(dataif.rover_covsel)
    for sub_dir in ["data", "submodels"]:
        (dataif.rover_covsel / sub_dir).mkdir(parents=True)

    # Create rover subsets
    get_rover_covsel_submodels(dataif.experiment, save_file=True)


def _initialize_regmod_smooth_results(dataif: DataInterface) -> None:
    # Initialize directories
    if dataif.regmod_smooth.exists():
        shutil.rmtree(dataif.regmod_smooth)
    dataif.regmod_smooth.mkdir(parents=True)


def _initialize_swimr_results(dataif: DataInterface) -> None:
    """Initialize swimr results."""
    # Initialize directories
    if dataif.swimr.exists():
        shutil.rmtree(dataif.swimr)
    for sub_dir in ["data", "submodels"]:
        (dataif.swimr / sub_dir).mkdir(parents=True)

    # Create swimr parameters and subsets
    get_swimr_submodels(dataif.experiment, save_files=True)


def _initialize_weave_results(dataif: DataInterface) -> None:
    """Initialize weave results."""

    # Initialize directories
    if dataif.weave.exists():
        shutil.rmtree(dataif.weave)
    (dataif.weave / "submodels").mkdir(parents=True)

    # Create weave parameters and subsets
    get_weave_submodels(dataif.experiment, save_files=True)


def _initialize_ensemble_results(dataif: DataInterface) -> None:
    """Initialize ensemble results."""

    # Initialize directory
    if dataif.ensemble.exists():
        shutil.rmtree(dataif.ensemble)
    dataif.ensemble.mkdir(parents=True)

    # Create ensemble subsets
    settings = dataif.load_settings()
    if "groupby" in settings["ensemble"]:
        Subsets(
            "ensemble", settings["ensemble"], get_ensemble_input(settings)
        ).subsets.to_csv(dataif.ensemble / "subsets.csv", index=False)


def main() -> None:
    fire.Fire(initialize_results)
