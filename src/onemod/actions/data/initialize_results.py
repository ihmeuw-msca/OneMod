"""Initialize onemod stage results."""

import shutil

import fire
from pplkit.data.interface import DataInterface

from onemod.utils import (
    get_ensemble_submodels,
    get_handle,
    get_rover_covsel_submodels,
    get_weave_submodels,
)


def initialize_results(directory: str, stages: list[str]) -> None:
    stage_init_map: dict[str, callable] = {
        "rover_covsel": _initialize_rover_covsel_results,
        "regmod_smooth": _initialize_regmod_smooth_results,
        "weave": _initialize_weave_results,
        "ensemble": _initialize_ensemble_results,
    }

    dataif, config = get_handle(directory)

    # ETL the input data into parquet format.
    # More compressible, faster IO, allows for partitioning
    raw_input_path = config.input_path
    data = dataif.load(raw_input_path)

    # subset data with ids
    if config.id_subsets:
        data = data.query(
            " & ".join(
                [
                    f"{key}.isin({value})"
                    for key, value in config.id_subsets.items()
                ]
            )
        ).reset_index(drop=True)

    # Saves to $directory/data/data.parquet
    dataif.dump_data(data)

    for stage in stages:
        stage_init_map[stage](dataif)


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
    get_ensemble_submodels(dataif.experiment, save_file=True)


def main() -> None:
    fire.Fire(initialize_results)
