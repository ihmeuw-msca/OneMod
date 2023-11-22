from pathlib import Path

from onemod.data.initialize_results import initialize_results
from onemod.models.rover_covsel_model import rover_covsel_model


def test_rover_model(sample_input_data, temporary_directory):
    initialize_results(experiment_dir=temporary_directory, stages=['rover_covsel'])
    rover_covsel_model(experiment_dir=temporary_directory, submodel_id='subset0')

    expected_data_path = Path(
        temporary_directory / "results" / "rover_covsel"/ "submodels" / "subset0" / "summary.csv"
    )
    assert expected_data_path.exists()
