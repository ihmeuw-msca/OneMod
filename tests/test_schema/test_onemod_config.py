from onemod.schema.models.api import (
    OneModConfig,
    RoverCovselConfiguration,
    RegmodSmoothConfiguration,
)


def test_onemod_config(sample_config_file):

    # No validation error raised
    parent_config = OneModConfig(**sample_config_file)

    # Assert we have an appropriate rover config and regmod config object
    assert hasattr(parent_config, 'rover_covsel')
    assert isinstance(parent_config.rover_covsel, RoverCovselConfiguration)

    assert hasattr(parent_config, 'regmod_smooth')
    assert isinstance(parent_config.regmod_smooth, RegmodSmoothConfiguration)