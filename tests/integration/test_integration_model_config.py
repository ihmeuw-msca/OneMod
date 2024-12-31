"""Test errors thrown if missing required model attributes."""

import pytest

from onemod.config import Config, RoverConfig, SpxmodConfig
from onemod.config.model_config.spxmod_config import SpxmodModelConfig

CONFIG_ITEMS = {
    "id_columns": ["sex_id", "year_id"],
    "model_type": "binomial",
    "observation_column": "obs",
    "prediction_column": "pred",
    "weights_column": "weights",
    "holdout_columns": ["holdout1", "holdout2"],
}

REQUIRED_ITEMS = {
    "rover": [
        "holdout_columns",
        "model_type",
        "observation_column",
        "weights_column",
    ],
    "spxmod": [
        "id_columns",
        "model_type",
        "observation_column",
        "prediction_column",
        "weights_column",
    ],
}

STAGE_DICT = {
    "rover": RoverConfig(cov_exploring={"cov1", "cov2"}),
    "spxmod": SpxmodConfig(xmodel=SpxmodModelConfig(variables=[])),
}


@pytest.mark.parametrize("stage", ["rover", "spxmod"])
@pytest.mark.parametrize("is_none", [True, False])
def test_config_forward(stage, is_none):
    stage_config = STAGE_DICT[stage]
    if is_none:
        pipeline_config = Config(
            **{item: None for item in REQUIRED_ITEMS[stage]}
        )
    else:
        pipeline_config = Config()
    missing = list(REQUIRED_ITEMS[stage])

    for item in REQUIRED_ITEMS[stage]:
        with pytest.raises(AttributeError) as e:
            stage_config.add_pipeline_config(pipeline_config)
        assert str(e.value) == f"Missing required config items: {missing}"

        pipeline_config[item] = CONFIG_ITEMS[item]
        missing.remove(item)


@pytest.mark.parametrize("stage", ["rover", "spxmod"])
@pytest.mark.parametrize("is_none", [True, False])
def test_config_backward(stage, is_none):
    stage_config = STAGE_DICT[stage]
    pipeline_config = Config(**CONFIG_ITEMS)
    missing = []

    for item in REQUIRED_ITEMS[stage]:
        if is_none:
            pipeline_config[item] = None
        else:
            delattr(pipeline_config, item)
        missing.append(item)

        with pytest.raises(AttributeError) as e:
            stage_config.add_pipeline_config(pipeline_config)
        assert str(e.value) == f"Missing required config items: {missing}"
