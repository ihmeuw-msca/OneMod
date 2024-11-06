import pytest

from onemod.datatools.config_interface import ConfigInterface


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

    assert len(configif.dirs) == 0

    configif.add_dir("config", tmp_path)

    assert len(configif.dirs) == 1
    assert "config" in configif.dirs
    assert configif.dirs["config"] == tmp_path


def test_add_dir_exist_ok(tmp_path):
    configif = ConfigInterface(config=tmp_path)

    with pytest.raises(ValueError):
        configif.add_dir("config", tmp_path)

    configif.add_dir("config", tmp_path, exist_ok=True)


def test_remove_dir(tmp_path):
    configif = ConfigInterface(config=tmp_path)

    assert len(configif.dirs) == 1

    configif.remove_dir("config")
    assert len(configif.dirs) == 0
