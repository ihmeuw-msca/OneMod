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
        modified_data.pop("weights")
        RoverConfiguration(**modified_data)

    # Try a non recognized model type
    with pytest.raises(ValidationError):
        modified_data["model_type"] = "not_a_model_type"
        RoverConfiguration(**modified_data)
