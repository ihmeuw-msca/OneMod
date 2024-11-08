import pytest

from onemod.fsutils.config_interface import ConfigInterface


@pytest.fixture
def sample_config():
    return {"param1": "value1", "param2": [1, 2, 3]}


@pytest.mark.parametrize("fextn", [".json", ".yaml", ".pkl"])
def test_config_interface(sample_config, fextn, tmp_path):
    configif = ConfigInterface(config=tmp_path)

    file_name = f"config{fextn}"
    configif.dump(sample_config, file_name, key="config")

    loaded_config = configif.load(file_name, key="config")

    assert loaded_config == sample_config


def test_add_dir(tmp_path):
    configif = ConfigInterface()

    assert len(configif.paths) == 0

    configif.add_path("config", tmp_path)

    assert len(configif.paths) == 1
    assert "config" in configif.paths
    assert configif.paths["config"] == tmp_path


def test_add_dir_exist_ok(tmp_path):
    configif = ConfigInterface(config=tmp_path)

    with pytest.raises(ValueError):
        configif.add_path("config", tmp_path)

    configif.add_path("config", tmp_path, exist_ok=True)


def test_remove_dir(tmp_path):
    configif = ConfigInterface(config=tmp_path)

    assert len(configif.paths) == 1

    configif.remove_path("config")
    assert len(configif.paths) == 0
