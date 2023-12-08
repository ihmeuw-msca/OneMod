from pathlib import Path
import yaml

# from onemod.data.initialize_results import initialize_results
from onemod.models.regmod_smooth_model import regmod_smooth_model


def test_regmod_smooth(temporary_directory, sample_input_data):

    # Mock the rover output - selected_covs.yaml
    # initialize_results(experiment_dir=temporary_directory,
    #                    stages=["rover_covsel", "regmod_smooth"])
    selected_covs = ["cov1", "cov2"]
    yaml_path = temporary_directory / "results" / "rover_covsel" / "selected_covs.yaml"
    if not yaml_path.parent.exists():
        yaml_path.parent.mkdir(parents=True)
    with open(yaml_path, "w") as f:
        yaml.dump(selected_covs, f)

    regmod_smooth_model(experiment_dir=temporary_directory)

    expected_data_path = Path(
        temporary_directory / "results" / "regmod_smooth" / "predictions.parquet"
    )
    assert expected_data_path.exists()
