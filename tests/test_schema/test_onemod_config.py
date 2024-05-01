import pytest
from pydantic import ValidationError

from onemod.schema import OneModConfig
from onemod.schema.stages import RoverCovselConfig, SPxModConfig


def test_onemod_config(sample_config_file):
    # No validation error raised
    parent_config = OneModConfig(**sample_config_file)

    # Assert we have an appropriate rover config and regmod config object
    assert hasattr(parent_config, "rover_covsel")
    assert isinstance(parent_config.rover_covsel, RoverCovselConfig)

    assert hasattr(parent_config, "spxmod")
    assert isinstance(parent_config.spxmod, SPxModConfig)

    # Try a non recognized model type
    with pytest.raises(ValidationError):
        modified_data = sample_config_file.copy()
        modified_data["mtype"] = "not_a_model"
        OneModConfig(**modified_data)
