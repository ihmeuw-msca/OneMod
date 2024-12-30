"""Test stage config class."""

import pytest

from onemod.config import Config, StageConfig


@pytest.fixture(scope="function")
def pipeline_config():
    return Config(
        pipeline_key="pipeline_value",
        shared_key="pipeline_shared_value",
        none_key=None,
    )


@pytest.fixture(scope="function")
def stage_config(pipeline_config):
    stage_config = StageConfig(
        stage_key="stage_value", shared_key="stage_shared_value", none_key=None
    )
    stage_config.add_pipeline_config(pipeline_config)
    return stage_config


@pytest.mark.parametrize("from_config", [True, False])
def test_pipeline_config(pipeline_config, from_config):
    stage_config = StageConfig(
        stage_key="stage_value", shared_key="stage_shared_value", none_key=None
    )
    if from_config:
        stage_config.add_pipeline_config(pipeline_config)
    else:  # from dictionary
        stage_config.add_pipeline_config(pipeline_config.model_dump())
    assert isinstance(stage_config._pipeline_config, Config)
    assert stage_config._pipeline_config["pipeline_key"] == "pipeline_value"
    assert (
        stage_config._pipeline_config["shared_key"] == "pipeline_shared_value"
    )


def test_contains(stage_config):
    assert "pipeline_key" in stage_config
    assert "stage_key" in stage_config
    assert "shared_key" in stage_config
    assert "none_key" not in stage_config
    assert "dummy_key" not in stage_config


def test_stage_contains(stage_config):
    assert stage_config.stage_contains("pipeline_key") is False
    assert stage_config.stage_contains("stage_key") is True
    assert stage_config.stage_contains("shared_key") is True
    assert stage_config.stage_contains("none_key") is False
    assert stage_config.stage_contains("dummy_key") is False


def test_pipeline_contains(stage_config):
    assert stage_config.pipeline_contains("pipeline_key") is True
    assert stage_config.pipeline_contains("stage_key") is False
    assert stage_config.pipeline_contains("shared_key") is True
    assert stage_config.pipeline_contains("none_key") is False
    assert stage_config.pipeline_contains("dummy_key") is False


def test_get(stage_config):
    assert stage_config.get("pipeline_key") == "pipeline_value"
    assert stage_config.get("stage_key") == "stage_value"
    assert stage_config.get("shared_key") == "stage_shared_value"


@pytest.mark.parametrize("key", ["none_key", "dummy_key"])
def test_get_default(stage_config, key):
    assert stage_config.get(key) is None
    assert stage_config.get(key, "default") == "default"


def test_get_from_stage(stage_config):
    assert stage_config.get_from_stage("stage_key") == "stage_value"
    assert stage_config.get_from_stage("shared_key") == "stage_shared_value"


@pytest.mark.parametrize("key", ["pipeline_key", "none_key", "dummy_key"])
def test_get_from_stage_default(stage_config, key):
    assert stage_config.get_from_stage(key) is None
    assert stage_config.get_from_stage(key, "default") == "default"


def test_get_from_pipeline(stage_config):
    assert stage_config.get_from_pipeline("pipeline_key") == "pipeline_value"
    assert (
        stage_config.get_from_pipeline("shared_key") == "pipeline_shared_value"
    )


@pytest.mark.parametrize("key", ["stage_key", "none_key", "dummy_key"])
def test_get_from_pipeline_default(stage_config, key):
    assert stage_config.get_from_pipeline(key) is None
    assert stage_config.get_from_pipeline(key, "default") == "default"


def test_getitem(stage_config):
    assert stage_config["pipeline_key"] == "pipeline_value"
    assert stage_config["stage_key"] == "stage_value"
    assert stage_config["shared_key"] == "stage_shared_value"


@pytest.mark.parametrize("key", ["none_key", "dummy_key"])
def test_getitem_error(stage_config, key):
    with pytest.raises(KeyError) as e:
        stage_config[key]
        assert str(e) == f"'Invalid config item: {key}'"


@pytest.mark.parametrize("key", ["stage_key", "new_key"])
def test_setitem_stage(stage_config, key):
    stage_config[key] = "new_value"
    assert key in stage_config
    assert stage_config.stage_contains(key) is True
    assert stage_config.pipeline_contains(key) is False
    assert stage_config.get(key) == "new_value"
    assert stage_config.get_from_stage(key) == "new_value"
    assert stage_config.get_from_pipeline(key) is None
    assert stage_config[key] == "new_value"


@pytest.mark.parametrize("key", ["pipeline_key", "new_key"])
def test_setitem_pipeline(stage_config, pipeline_config, key):
    pipeline_config[key] = "new_value"
    assert key in stage_config
    assert stage_config.stage_contains(key) is False
    assert stage_config.pipeline_contains(key) is True
    assert stage_config.get(key) == "new_value"
    assert stage_config.get_from_stage(key) is None
    assert stage_config.get_from_pipeline(key) == "new_value"
    assert stage_config[key] == "new_value"


def test_repr(stage_config):
    assert (
        repr(stage_config)
        == "StageConfig(pipeline_key='pipeline_value', shared_key='stage_shared_value', stage_key='stage_value')"
    )
