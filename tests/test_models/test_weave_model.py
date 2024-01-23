import numpy as np
from pathlib import Path

from onemod.actions.models.weave_model import weave_model
from onemod.utils import get_weave_submodels


def test_weave_model(sample_input_data, temporary_directory):
    # Mock a regmod smooth
    regmod_smooth_path = (
        temporary_directory / "results" / "regmod_smooth" / "predictions.parquet"
    )
    if not regmod_smooth_path.parent.exists():
        regmod_smooth_path.parent.mkdir(parents=True)

    sample_input_data["pred_rate"] = np.random.rand(len(sample_input_data))
    sample_input_data["residual"] = (
        sample_input_data["obs_rate"] - sample_input_data["pred_rate"]
    )
    # Assume a binomial model for this rates data
    sample_input_data["residual_se"] = 1 / np.sqrt(
        sample_input_data["pred_rate"] * (1 - sample_input_data["pred_rate"])
    )

    sample_input_data.to_parquet(regmod_smooth_path)

    # Initialize weave directories and parameter files
    get_weave_submodels(temporary_directory, save_files=True)
    submodel_id = "model1_param0_subset0_holdout1_batch0"
    weave_model(experiment_dir=temporary_directory, submodel_id=submodel_id)
    expected_data_path = Path(
        temporary_directory
        / "results"
        / "weave"
        / "submodels"
        / f"{submodel_id}.parquet"
    )
    assert expected_data_path.exists()
