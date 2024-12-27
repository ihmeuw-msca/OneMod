import json
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel
from tests.helpers.utils import assert_equal_unordered

from onemod.dtypes import UniqueList
from onemod.serialization import deserialize, serialize


class ExampleSubModel(BaseModel):
    """Example sub-model."""

    name: str
    value: int


class ExampleModel(BaseModel):
    """Example model with a variety of attribute types."""

    name: str
    model: ExampleSubModel
    ids: list[int]
    shape: tuple
    config: dict
    groupby: UniqueList[str]
    items: dict[str, Path]


@pytest.fixture
def tmp_json_file(tmp_path):
    return tmp_path / "config.json"


@pytest.fixture
def tmp_yaml_file(tmp_path):
    return tmp_path / "config.yaml"


@pytest.fixture
def test_model() -> ExampleModel:
    return ExampleModel(
        name="test",
        model=ExampleSubModel(name="base", value=1),
        ids=[1, 2, 3],
        shape=(5, 10),
        config={"key": "value"},
        groupby={"a", "b", "c"},
        items={"path1": Path("data1.parquet"), "path2": Path("data2.csv")},
    )


@pytest.fixture
def test_model_dict_repr() -> dict:
    return {
        "name": "test",
        "model": {"name": "base", "value": 1},
        "ids": [1, 2, 3],
        "shape": [5, 10],
        "config": {"key": "value"},
        "groupby": ["a", "b", "c"],
        "items": {"path1": "data1.parquet", "path2": "data2.csv"},
    }


@pytest.fixture
def test_dict() -> dict:
    return {
        "name": "test_pipeline",
        "config": {"param1": 1, "param2": 2},
        "ids": [1, 2, 3],
        "dependencies": ["task1", "task2"],
    }


@pytest.mark.unit
def test_serialize_dict_json(test_dict, tmp_json_file):
    """Test serializing a dictionary to JSON."""
    serialize(test_dict, tmp_json_file)
    with open(tmp_json_file) as f:
        data = json.load(f)
    assert_equal_unordered(data, test_dict)


@pytest.mark.unit
def test_serialize_dict_yaml(test_dict, tmp_yaml_file):
    """Test serializing a dictionary to YAML."""
    serialize(test_dict, tmp_yaml_file)
    with open(tmp_yaml_file) as f:
        data = yaml.safe_load(f)
    assert_equal_unordered(data, test_dict)


@pytest.mark.unit
def test_serialize_model_json(test_model, test_model_dict_repr, tmp_json_file):
    """Test serializing a Pydantic model to JSON."""
    serialize(test_model, tmp_json_file)
    with open(tmp_json_file) as f:
        data = json.load(f)
    assert_equal_unordered(data, test_model_dict_repr)


@pytest.mark.unit
def test_serialize_model_yaml(test_model, test_model_dict_repr, tmp_yaml_file):
    """Test serializing a Pydantic model to YAML."""
    serialize(test_model, tmp_yaml_file)
    with open(tmp_yaml_file) as f:
        data = yaml.safe_load(f)
    assert_equal_unordered(data, test_model_dict_repr)


@pytest.mark.unit
def test_serialize_list_of_dicts_json(test_dict, tmp_json_file):
    """Test serializing a list of dictionaries to JSON."""
    dicts = [test_dict, test_dict]
    serialize(dicts, tmp_json_file)
    with open(tmp_json_file) as f:
        data = json.load(f)
    assert_equal_unordered(data, [test_dict, test_dict])


@pytest.mark.unit
def test_serialize_list_of_dicts_yaml(test_dict, tmp_yaml_file):
    """Test serializing a list of dictionaries to YAML."""
    dicts = [test_dict, test_dict]
    serialize(dicts, tmp_yaml_file)
    with open(tmp_yaml_file) as f:
        data = yaml.safe_load(f)
    assert_equal_unordered(data, [test_dict, test_dict])


@pytest.mark.unit
def test_serialize_list_of_models_json(
    test_model, test_model_dict_repr, tmp_json_file
):
    """Test serializing a list of Pydantic models to JSON."""
    models = [test_model, test_model]
    serialize(models, tmp_json_file)
    with open(tmp_json_file) as f:
        data = json.load(f)
    assert_equal_unordered(data, [test_model_dict_repr, test_model_dict_repr])


@pytest.mark.unit
def test_serialize_list_of_models_yaml(
    test_model, test_model_dict_repr, tmp_yaml_file
):
    """Test serializing a list of Pydantic models to YAML."""
    models = [test_model, test_model]
    serialize(models, tmp_yaml_file)
    with open(tmp_yaml_file) as f:
        data = yaml.safe_load(f)
    assert_equal_unordered(data, [test_model_dict_repr, test_model_dict_repr])


@pytest.mark.unit
def test_deserialize_json(test_dict, tmp_json_file):
    """Test deserializing a JSON file into a dictionary."""
    with open(tmp_json_file, "w") as f:
        json.dump(test_dict, f)
    data = deserialize(tmp_json_file)
    assert_equal_unordered(data, test_dict)


@pytest.mark.unit
def test_deserialize_yaml(test_dict, tmp_yaml_file):
    """Test deserializing a YAML file into a dictionary."""
    with open(tmp_yaml_file, "w") as f:
        yaml.safe_dump(test_dict, f)
    data = deserialize(tmp_yaml_file)
    assert_equal_unordered(data, test_dict)


@pytest.mark.unit
def test_unsupported_format(tmp_path):
    """Test that an unsupported file format raises a ValueError."""
    unsupported_file = tmp_path / "unsupported.txt"
    with pytest.raises(ValueError, match="Unsupported file format"):
        serialize({}, unsupported_file)
