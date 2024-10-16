import json
import yaml
from pathlib import Path
from typing import List
from yaml import Node, SafeDumper

from pydantic import BaseModel


def _yaml_path_representer(dumper: SafeDumper, data: Path) -> Node:
    return dumper.represent_str(str(data))


def _yaml_set_representer(dumper: SafeDumper, data: set) -> Node:
    return dumper.represent_list(list(data))


yaml.add_multi_representer(Path, _yaml_path_representer, SafeDumper)
yaml.add_representer(set, _yaml_set_representer, SafeDumper)


def deserialize(filepath: Path | str) -> dict:
    """Load a configuration from a YAML or JSON file."""
    file_format = _get_file_format(filepath)
    with open(filepath, "r") as file:
        if file_format == "json":
            return json.load(file)
        elif file_format == "yaml":
            return yaml.safe_load(file)


def serialize(
    obj: BaseModel | dict | List[BaseModel] | List[dict], filepath: Path | str
) -> None:
    """Save a Pydantic model, dict, or list of these to a YAML or JSON file."""
    file_format = _get_file_format(filepath)
    if file_format not in {"json", "yaml"}:
        raise ValueError(f"Unsupported file format: {file_format}")

    if isinstance(obj, BaseModel):
        data = obj.model_dump()
    elif isinstance(obj, dict):
        data = obj.copy()
        for key, value in data.items():
            if isinstance(value, set):
                data[key] = list(value)
    elif isinstance(obj, list):
        if all(isinstance(item, BaseModel) for item in obj):
            data = [item.model_dump() for item in obj]
        elif all(isinstance(item, dict) for item in obj):
            data = []
            for item in obj:
                item_copy = item.copy()
                for key, value in item_copy.items():
                    if isinstance(value, set):
                        item_copy[key] = list(value)
                data.append(item_copy)
        else:
            raise TypeError(
                "All items in the list must be either BaseModel instances or dicts."
            )
    else:
        raise TypeError(
            "Object to serialize must be a BaseModel, dict, or a list of these."
        )

    with open(filepath, "w") as file:
        if file_format == "json":
            json.dump(data, file, indent=4, default=_json_serializer)
        elif file_format == "yaml":
            yaml.safe_dump(data, file, default_flow_style=False)


def _get_file_format(filepath: Path | str) -> str:
    """Detect the file format based on the file extension."""
    file_extension = Path(filepath).suffix.lower()
    if file_extension in {".json"}:
        return "json"
    elif file_extension in {".yaml", ".yml"}:
        return "yaml"
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. Supported formats are: .json, .yaml, .yml"
        )


def _json_serializer(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
