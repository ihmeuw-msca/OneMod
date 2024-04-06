from pathlib import Path

from onemod.actions.data.initialize_results import initialize_results
from onemod.actions.models.rover_covsel_model import rover_covsel_model


def test_rover_model(temporary_directory):
    initialize_results(directory=temporary_directory, stages=["rover_covsel"])
    rover_covsel_model(directory=temporary_directory, submodel_id="subset0")

    expected_data_path = Path(
        temporary_directory
        / "results"
        / "rover_covsel"
        / "submodels"
        / "subset0"
        / "summary.csv"
    )
    assert expected_data_path.exists()
