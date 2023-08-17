"""Initialize onemod stage results."""
import shutil

import fire

from onemod.utils import (
    get_ensemble_input,
    get_rover_covsel_submodels,
    get_swimr_submodels,
    get_weave_submodels,
    Subsets,
    get_data_interface,
)


def initialize_rover_covsel_results(experiment_dir: str) -> None:
    """Initialize rover results."""
    dataif = get_data_interface(experiment_dir)

    # Initialize directories
    if dataif.rover_covsel.exists():
        shutil.rmtree(dataif.rover_covsel)
    for sub_dir in ["data", "submodels"]:
        (dataif.rover_covsel / sub_dir).mkdir(parents=True)

    # Create rover subsets
    get_rover_covsel_submodels(experiment_dir, save_file=True)


def initialize_regmod_smooth_results(experiment_dir: str) -> None:
    dataif = get_data_interface(experiment_dir)

    # Initialize directories
    if dataif.regmod_smooth.exists():
        shutil.rmtree(dataif.regmod_smooth)
    dataif.regmod_smooth.mkdir(parents=True)


def initialize_swimr_results(experiment_dir: str) -> None:
    """Initialize swimr results."""
    dataif = get_data_interface(experiment_dir)

    # Initialize directories
    if dataif.swimr.exists():
        shutil.rmtree(dataif.swimr)
    for sub_dir in ["data", "submodels"]:
        (dataif.swimr / sub_dir).mkdir(parents=True)

    # Create swimr parameters and subsets
    get_swimr_submodels(experiment_dir, save_files=True)


def initialize_weave_results(experiment_dir: str) -> None:
    """Initialize weave results."""
    dataif = get_data_interface(experiment_dir)

    # Initialize directories
    if dataif.weave.exists():
        shutil.rmtree(dataif.weave)
    (dataif.weave / "submodels").mkdir(parents=True)

    # Create weave parameters and subsets
    get_weave_submodels(experiment_dir, save_files=True)


def initialize_ensemble_results(experiment_dir: str) -> None:
    """Initialize ensemble results."""
    dataif = get_data_interface(experiment_dir)

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
    fire.Fire(
        {
            "rover_covsel": initialize_rover_covsel_results,
            "regmod_smooth": initialize_regmod_smooth_results,
            "swimr": initialize_swimr_results,
            "weave": initialize_weave_results,
            "ensemble": initialize_ensemble_results,
        }
    )
