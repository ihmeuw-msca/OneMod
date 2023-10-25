import pandas as pd
from pydantic import BaseModel, ValidationError

from onemod.schema.config import ParentConfiguration
from onemod.utils import get_data_interface


def validate_config(
    stages: list[str],
    directory: str,
    config: ParentConfiguration | None = None,
) -> None:
    """Validate the configuration file according to the expected schema.

    Either a configuration or a directory to load that configuration from must be provided.
    """

    dataif = get_data_interface(directory)

    if not config:

        settings = dataif.load_settings()

        # Validation error raised if config fails
        config = ParentConfiguration(**settings)

    # Validate against the dataset
    dataset = dataif.load(config.input_path)
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
