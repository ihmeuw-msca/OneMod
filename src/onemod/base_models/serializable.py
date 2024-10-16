from pathlib import Path
from typing import Type, TypeVar
import json
import yaml
from pydantic import BaseModel

T = TypeVar('T', bound='SerializableModel')


class SerializableModel(BaseModel):
    def to_json(self, filepath: Path | str, exclude_none: bool = True) -> None:
        """Serialize the model to a JSON file."""
        filepath = Path(filepath)
        with filepath.open('w') as f:
            f.write(self.model_dump_json(indent=4, exclude_none=exclude_none))

    @classmethod
    def from_json(cls: Type[T], filepath: Path | str) -> T:
        """Deserialize the model from a JSON file."""
        filepath = Path(filepath)
        with filepath.open('r') as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_yaml(self, filepath: Path | str, exclude_none: bool = True) -> None:
        """Serialize the model to a YAML file."""
        filepath = Path(filepath)
        data = self.model_dump(exclude_none=exclude_none)
        with filepath.open('w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls: Type[T], filepath: Path | str) -> T:
        """Deserialize the model from a YAML file."""
        filepath = Path(filepath)
        with filepath.open('r') as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
