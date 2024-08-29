from pathlib import Path

import yaml

from onemod.actions.data.initialize_results import _initialize_rover_covsel_results, _initialize_spxmod_results
from onemod.actions.models.spxmod_model import spxmod_model


def test_spxmod(temporary_directory):
    # Mock the rover output - selected_covs.yaml
    _initialize_rover_covsel_results(dataif=temporary_directory)
    _initialize_spxmod_results(dataif=temporary_directory)
    selected_covs = ["cov1", "cov2"]
    yaml_path = (
        temporary_directory / "results" / "rover_covsel" / "selected_covs.yaml"
    )
    if not yaml_path.parent.exists():
        yaml_path.parent.mkdir(parents=True)
    with open(yaml_path, "w") as f:
        yaml.dump(selected_covs, f)

    spxmod_model.task_function(directory=temporary_directory, submodel_id="0")

    expected_data_path = Path(
        temporary_directory / "results" / "spxmod" / "predictions.parquet"
    )
    assert expected_data_path.exists()
