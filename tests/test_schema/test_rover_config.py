import pytest
from onemod.schema.models.api import RoverCovselConfig
from pydantic import ValidationError


def test_rover_config(sample_config):
    """Test rover model validation and configuration."""

    # Check that inheritance works properly
    rover_config = sample_config.rover_covsel
    rover_config.inherit()
    assert rover_config.mtype == sample_config.mtype
    assert rover_config.max_attempts == sample_config.max_attempts

    # If a required key is missing, raise a Validation Error
    with pytest.raises(ValidationError):
        modified_data = sample_config["rover_covsel"].model_dump()
        modified_data["rover"].pop("weights")
        RoverCovselConfig(**modified_data)
