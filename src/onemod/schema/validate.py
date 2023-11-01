import pandas as pd
from pydantic import BaseModel, ValidationError

from onemod.schema.models.parent_config import ParentConfiguration
from onemod.utils import get_handle


def validate_config(directory: str, stages: list[str]) -> None:
    """Validate the configuration file according to the expected schema."""

    dataif, config = get_handle(directory)

    # Validate against the dataset
    dataset = dataif.load_data()
    errors = []

    if "rover_covsel" in stages:
        rover_errors = validate_rover_config(config, dataset)
        errors.extend(rover_errors)

    if "regmod_smooth" in stages:
        regmod_errors = validate_regmod_config(config, dataset)
        errors.extend(regmod_errors)

    if "weave" in stages:
        weave_errors = validate_weave_config(config, dataset)
        errors.extend(weave_errors)

    if "swimr" in stages:
        swimr_errors = validate_swimr_config(config, dataset)
        errors.extend(swimr_errors)

    if "ensemble" in stages:
        ensemble_errors = validate_ensemble_config(config, dataset)
        errors.extend(ensemble_errors)

    if any(errors):
        raise ValidationError(f"Following errors were raised: {errors}")


def validate_rover_config(config: BaseModel, dataset: pd.DataFrame) -> list[str]:
    return []  # TODO


def validate_regmod_config(config: BaseModel, dataset: pd.DataFrame) -> list[str]:
    return []  # TODO


def validate_weave_config(config: BaseModel, dataset: pd.DataFrame) -> list[str]:
    return []  # TODO


def validate_swimr_config(config: BaseModel, dataset: pd.DataFrame) -> list[str]:
    return []  # TODO


def validate_ensemble_config(config: BaseModel, dataset: pd.DataFrame) -> list[str]:
    return []  # TODO
