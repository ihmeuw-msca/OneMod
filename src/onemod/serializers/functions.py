import json
from pathlib import Path
from typing import Union
import yaml

from pydantic import BaseModel

def _get_file_format(file_path: Union[str, Path]) -> str:
    """Detect the file format based on the file extension."""
    if str(file_path).endswith('.json'):
        return 'json'
    elif str(file_path).endswith('.yaml') or str(file_path).endswith('.yml'):
        return 'yaml'
    raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

def deserialize(file_path: Union[str, Path]) -> dict:
    """Load a configuration from a YAML or JSON file."""
    file_format = _get_file_format(file_path)
    with open(file_path, 'r') as file:
        if file_format == 'json':
            return json.load(file)
        elif file_format == 'yaml':
            return yaml.safe_load(file)

def serialize(model: Union[dict, BaseModel], file_path: Union[str, Path]) -> None:
    """Save to a YAML or JSON file."""
    file_format = _get_file_format(file_path)

    if isinstance(model, dict):
        for key, value in model.items():
            if isinstance(value, set):
                model[key] = list(value)
    
    with open(file_path, 'w') as file:
        if file_format == 'json':
            if isinstance(model, BaseModel):
                file.write(model.model_dump_json(indent=4, exclude_none=True, serialize_as_any=True))
            else:
                json.dump(model, file, indent=4)
        elif file_format == 'yaml':
            if isinstance(model, BaseModel):
                yaml.safe_dump(model.model_dump(), file, default_flow_style=False)
            else:
                yaml.safe_dump(model, file, default_flow_style=False)
