from pathlib import Path

import numpy as np

from onemod.actions.models.weave_model import weave_model
from onemod.utils import get_weave_submodels


def test_weave_model(sample_input_data, temporary_directory):
    # Mock a spxmod
    spxmod_path = (
        temporary_directory / "results" / "spxmod" / "predictions.parquet"
    )
    if not spxmod_path.parent.exists():
        spxmod_path.parent.mkdir(parents=True)

    sample_input_data["pred_rate"] = np.random.rand(len(sample_input_data))
    sample_input_data["residual"] = (
        sample_input_data["obs_rate"] - sample_input_data["pred_rate"]
    )
    # Assume a binomial model for this rates data
    sample_input_data["residual_se"] = 1 / np.sqrt(
        sample_input_data["pred_rate"] * (1 - sample_input_data["pred_rate"])
    )

    sample_input_data.to_parquet(spxmod_path)

    # Initialize weave directories and parameter files
    get_weave_submodels(temporary_directory, save_files=True)
    submodel_id = "model1_param0_subset0_holdout1_batch0"
    weave_model(directory=temporary_directory, submodel_id=submodel_id)
    expected_data_path = Path(
        temporary_directory
        / "results"
        / "weave"
        / "submodels"
        / f"{submodel_id}.parquet"
    )
    assert expected_data_path.exists()
