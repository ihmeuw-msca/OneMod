import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel
from yaml import Node, SafeDumper

T = TypeVar("T", bound=BaseModel)


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
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


def serialize(obj: T | dict | list[T | dict], filepath: Path | str) -> None:
    """Save a Pydantic model, dict, or list of these to a YAML or JSON file."""
    file_format = _get_file_format(filepath)
    data = _normalize(obj)

    with open(filepath, "w") as file:
        if file_format == "json":
            json.dump(data, file, indent=4, cls=_CustomJSONEncoder)
        elif file_format == "yaml":
            yaml.safe_dump(data, file, default_flow_style=False)


class _CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle Path, set, and other unsupported types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)


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


def _normalize(obj: Any) -> Any:
    """Convert Pydantic models and sets to serializable forms."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj
