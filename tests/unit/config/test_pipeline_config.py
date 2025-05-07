"""Test pipeline config class."""

import pytest

from onemod.config import Config


@pytest.fixture(scope="function")
def config():
    return Config(key="value")


def test_contains(config):
    assert "key" in config
    assert "dummy_key" not in config


def test_get(config):
    assert config.get("key") == "value"


def test_get_default(config):
    assert config.get("dummy_key") is None
    assert config.get("dummy_key", "default") == "default"


def test_getitem(config):
    assert config["key"] == "value"


def test_getitem_error(config):
    with pytest.raises(KeyError) as e:
        config["dummy_key"]
        assert str(e) == "'Invalid config item: dummy_key'"


@pytest.mark.parametrize("key", ["key", "new_key"])
def test_setitem(config, key):
    config[key] = "new_value"
    assert key in config
    assert config.get(key) == "new_value"
    assert config[key] == "new_value"


def test_repr(config):
    assert repr(config) == "Config(key='value')"
