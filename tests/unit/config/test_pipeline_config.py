"""Test pipeline config class."""

import pytest

from onemod.config import Config


@pytest.fixture(scope="function")
def config():
    return Config(key="value")


def test_contains(config):
    assert "key" in config
    assert "dummy" not in config


def test_get(config):
    assert config.get("key") == "value"


@pytest.mark.parametrize("default", [None, "default"])
def test_get_default(config, default):
    if default is None:
        assert config.get("dummy") is None
    else:
        assert config.get("dummy", default) == default


def test_getitem(config):
    assert config["key"] == "value"


def test_getitem_error(config):
    with pytest.raises(KeyError) as e:
        config["dummy"]
        assert str(e) == "'Invalid config item: dummy'"


@pytest.mark.parametrize("key", ["key", "new_key"])
def test_item(config, key):
    config[key] = "new_value"
    assert key in config
    assert config.get(key) == "new_value"
    assert config[key] == "new_value"
