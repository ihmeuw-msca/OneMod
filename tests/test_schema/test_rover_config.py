from onemod.schema.config import RoverConfiguration
from pydantic import ValidationError
import pytest

def test_rover_config(sample_input_data, sample_config):
    """Test rover model validation and configuration."""

    # No validation error raised
    RoverConfiguration(**sample_config["rover_covsel"])

    # Try dropping a key
    with pytest.raises(ValidationError):
        modified_data = sample_config["rover_covsel"].copy()
        modified_data.pop("obs")
        RoverConfiguration(**modified_data)


def test_weave_config(sample_input_data, sample_config):
    """Test weave model validation and configuration."""

    # No validation error raised
    WeaveConfiguration(**sample_config["weave"])

    # Try dropping a key
    with pytest.raises(ValidationError):
        modified_data = sample_config["weave"].copy()
        modified_data.pop("obs")
        WeaveConfiguration(**modified_data)