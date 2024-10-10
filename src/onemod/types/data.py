from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Union

from polars import Boolean, DataFrame, Int64, Float64, String
from pydantic import BaseModel

from onemod.constraints import Constraint
from onemod.serializers import deserialize, serialize
from onemod.types.column_spec import ColumnSpec
from onemod.types.filepath import FilePath
from onemod.utils import DataIOHandler
from onemod.validation.error_handling import ValidationErrorCollector, handle_error


class Data(BaseModel):
    stage: str
    path: Union[Path | FilePath]
    format: str = "parquet"
    shape: Optional[tuple[int, int]] = None
    columns: Optional[Dict[str, ColumnSpec]] = None
    type_mapping: ClassVar[Dict[type, Any]] = {
        bool: Boolean,
        int: Int64,
        float: Float64,
        str: String,
    }
    
    @classmethod
    def from_dict(cls, data_dict: dict) -> 'Data':
        """Reconstruct a Data object from a dictionary."""
        return cls(
            stage=data_dict["stage"],
            path=Path(data_dict["path"]),
            format=data_dict.get("format", "parquet"),
            shape=tuple(data_dict["shape"]) if data_dict.get("shape") else None,
            columns={
                col_name: {
                    "type": col_spec.get("type", Any),
                    "constraints": [
                        Constraint.from_dict(constraint)
                        for constraint in col_spec.get("constraints", [])
                    ]
                }
                for col_name, col_spec in data_dict.get("columns", {}).items()
            } if data_dict.get("columns") else None
        )

    def to_dict(self) -> dict:
        """Convert the Data object to a dictionary."""
        return {
            "stage": self.stage,
            "path": str(self.path),
            "format": self.format,
            "shape": self.shape if self.shape else None,
            "columns": {
                col_name: {
                    "type": col_spec["type"].__name__,
                    "constraints": [constraint.to_dict() for constraint in col_spec.get("constraints", [])]
                }
                for col_name, col_spec in (self.columns or {}).items()
            } if self.columns else None
        }
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'Data':
        """Load a Data configuration from a YAML or JSON file."""
        config = deserialize(config_path)
        return cls(**config)

    def to_config(self, config_path: Union[str, Path]) -> None:
        """Save the current Data configuration to a YAML or JSON file."""
        serialize(self, config_path)
    
    def validate_metadata(self, collector: ValidationErrorCollector | None = None) -> None:
        """One-time validation for instance metadata."""
        if not self.path:
            handle_error(self.stage, "Data validation", ValueError,
                         "File path is required.", collector)
        else:
            if not self.path.exists():
                handle_error(self.stage, "Data validation", FileNotFoundError,
                             f"File {self.path} does not exist.", collector)
            if self.format not in DataIOHandler.supported_formats:
                handle_error(self.stage, "Data validation", ValueError,
                             f"Unsupported file format {self.format}.", collector)
        
        if self.shape:
            if not isinstance(self.shape, tuple) or len(self.shape) != 2:
                handle_error(self.stage, "Data validation", ValueError,
                             "Shape must be a tuple of (rows, columns).", collector)

    def validate_shape(self, data: DataFrame, collector: ValidationErrorCollector | None = None) -> None:
        """Validate the shape of the data."""
        if data.shape != self.shape:
            handle_error(self.stage, "Data validation", ValueError,
                         f"Expected DataFrame shape {self.shape}, got {data.shape}.", collector)

    def validate_data(self, data: DataFrame | None, collector: ValidationErrorCollector | None = None) -> None:
        """Validate the columns and shape of the data."""
        if data is None:
            try:
                data = DataIOHandler.read_data(self.path)
            except Exception as e:
                handle_error(self.stage, "Data validation", e.__class__, str(e), collector)
        
        if self.shape:
            self.validate_shape(data, collector)
        
        if self.columns:
            self.validate_columns(data, collector)
    
    def validate_columns(self, data: DataFrame, collector: ValidationErrorCollector | None = None) -> None:
        """Validate columns based on specified types and constraints."""
        for col_name, col_spec in self.columns.items():
            if col_name not in data.columns:
                handle_error(self.stage, "Data validation", ValueError,
                             f"Column '{col_name}' is missing from the data.", collector)
            
            expected_type = col_spec.get('type')
            constraints = col_spec.get('constraints', [])
            
            if expected_type:
                polars_type = self.type_mapping.get(expected_type)  # TODO: better ways to handle this?
                if not polars_type:
                    handle_error(self.stage, "Data validation", ValueError,
                                 f"Unsupported type {expected_type}.", collector)
                if data[col_name].dtype != polars_type:
                    handle_error(self.stage, "Data validation", ValueError,
                                 f"Column '{col_name}' must be of type {expected_type.__name__}.", collector)
            
            for constraint in constraints:
                constraint(data[col_name])
