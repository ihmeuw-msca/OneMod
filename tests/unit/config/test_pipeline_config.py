"""Test pipeline config class."""

import pytest

from onemod.config import Config


@pytest.fixture(scope="function")
def config():
    return Config(key="value", none_key=None)


def test_contains(config):
    assert "key" in config
    assert "none_key" not in config
    assert "dummy_key" not in config


def test_get(config):
    assert config.get("key") == "value"


@pytest.mark.parametrize("key", ["none_key", "dummy_key"])
def test_get_default(config, key):
    assert config.get(key) is None
    assert config.get(key, "default") == "default"


def test_getitem(config):
    assert config["key"] == "value"


@pytest.mark.parametrize("key", ["none_key", "dummy_key"])
def test_getitem_error(config, key):
    with pytest.raises(KeyError) as e:
        config[key]
        assert str(e) == f"'Invalid config item: {key}'"


@pytest.mark.parametrize("key", ["key", "new_key"])
def test_item(config, key):
    config[key] = "new_value"
    assert key in config
    assert config.get(key) == "new_value"
    assert config[key] == "new_value"
