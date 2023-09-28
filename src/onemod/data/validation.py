from pathlib import Path
from typing import Any, Union

from onemod.utils import get_data_interface


class ValidationError(Exception):
    pass


def validate_input_data(experiment_dir: Union[str, Path], stages: list[str]):
    """Validates the input dataset, and checks for some common errors in the data design.

    Note that this check is not comprehensive, i.e. passing this check does not guarantee
    that the data integrations are properly set up downstream.
    """

    interface = get_data_interface(experiment_dir)

    settings = interface.load_settings()
    dataset = interface.load_data()

    # Check: expected columns all present
    id_cols = fetch_key(settings, "col_id")
    missing_id_cols = set(id_cols) - set(dataset.columns)
    if any(missing_id_cols):
        raise ValidationError(f"Dataset is missing specified ID columns {missing_id_cols}")

    # Optional checks for specific stages in
    if "rover_covsel" in stages:
        _validate_data_for_rover_covsel(dataset, settings)

    if "regmod_smooth" in stages:
        _validate_data_for_regmod_smooth(dataset, settings)

    if "weave" in stages:
        _validate_data_for_weave(dataset, settings)

    if "swimr" in stages:
        _validate_data_for_swimr(dataset, settings)

    if "ensemble" in stages:
        _validate_data_for_ensemble(dataset, settings)


def _validate_data_for_ensemble(dataset, settings):
# Check: ensemble data has all expected columns
    ensemble_cols = fetch_key(settings, "ensemble", "columns")
    missing_ensemble_cols = set(ensemble_cols) - set(dataset.columns)
    if any(missing_ensemble_cols):
        raise ValidationError(
            f"Dataset is missing specified ensemble columns {missing_ensemble_cols}"
        )

    try:

"""
Business rules:
1. No data in a subset
  File "/mnt/share/homes/dhs2018/repos/OneMod/src/onemod/models/rover_covsel_model.py", line 59, in rover_covsel_model
    rover.fit(data=df_train, **settings["rover_covsel"]["Rover.fit"])
  File "/mnt/share/homes/dhs2018/repos/modrover/src/modrover/rover.py", line 174, in fit
    self._get_super_learner(
  File "/mnt/share/homes/dhs2018/repos/modrover/src/modrover/rover.py", line 439, in _get_super_learner
    df = self._get_learner_info(top_pct_score, top_pct_learner, coef_bounds)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/share/homes/dhs2018/repos/modrover/src/modrover/rover.py", line 486, in _get_learner_info
    df.loc[df["valid"], "weight"] = self._get_super_weights(
                                    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/share/homes/dhs2018/repos/modrover/src/modrover/rover.py", line 544, in _get_super_weights
    indices = scores >= scores[argsort[0]] * (1 - top_pct_score)
                               ~~~~~~~^^^
IndexError: index 0 is out of bounds for axis 0 with size 0


"""