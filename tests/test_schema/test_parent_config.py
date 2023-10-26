from onemod.schema.config import (
    OneModCFG,
    RegmodSmoothConfiguration,
    RoverConfiguration,
)


def test_parent_config(sample_input_data, sample_config):
    # No validation error raised
    parent_config = OneModCFG(**sample_config)

    # Assert we have an appropriate rover config object
    assert hasattr(parent_config, "rover_covsel")
    assert isinstance(parent_config.rover_covsel, RoverConfiguration)
    assert (
        parent_config.rover_covsel.model_type
        == sample_config["rover_covsel"]["model_type"]
    )

    assert hasattr(parent_config, "regmod_smooth")
    assert isinstance(parent_config.regmod_smooth, RegmodSmoothConfiguration)
