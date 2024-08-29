from pathlib import Path

import yaml

from onemod.actions.models.spxmod_model import spxmod_model


def test_spxmod(temporary_directory):
    # Mock the rover output - selected_covs.yaml
    # Mocks:
    # initialize_results(directory=temporary_directory,
    #                    stages=["rover_covsel", "spxmod"])
    selected_covs = ["cov1", "cov2"]
    yaml_path = (
        temporary_directory / "results" / "rover_covsel" / "selected_covs.yaml"
    )
    if not yaml_path.parent.exists():
        yaml_path.parent.mkdir(parents=True)
    with open(yaml_path, "w") as f:
        yaml.dump(selected_covs, f)

    # @ToDo Next call is broken, needs submodel_id but it is not provided
    spxmod_model.task_function(directory=temporary_directory)

    expected_data_path = Path(
        temporary_directory / "results" / "spxmod" / "predictions.parquet"
    )
    assert expected_data_path.exists()
