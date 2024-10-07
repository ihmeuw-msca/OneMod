from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Union

from polars import Boolean, DataFrame, Int64, Float64, String
from pydantic import BaseModel

from onemod.constraints import Constraint
from onemod.io.data_io_handler import DataIOHandler
from onemod.serializers import deserialize, serialize
from onemod.types.column_spec import ColumnSpec
from onemod.types.filepath import FilePath
from onemod.validation import collector as validation_collector


class Data(BaseModel):
    stage: str
    path: FilePath
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
            }
        )

    def to_dict(self) -> dict:
        """Convert the Data object to a dictionary."""
        return {
            "stage": self.stage,
            "path": str(self.path),
            "format": self.format,
            "shape": self.shape,
            "columns": {
                col_name: {
                    "type": col_spec["type"].__name__,
                    "constraints": [constraint.to_dict() for constraint in col_spec.get("constraints", [])]
                }
                for col_name, col_spec in (self.columns or {}).items()
            }
        }
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'Data':
        """Load a Data configuration from a YAML or JSON file."""
        config = deserialize(config_path)
        return cls(**config)

    def to_config(self, config_path: Union[str, Path]) -> None:
        """Save the current Data configuration to a YAML or JSON file."""
        serialize(self, config_path)

    @classmethod
    def use_validation(
        cls,
        columns: Optional[Dict[str, ColumnSpec]] = None,
        shape: Optional[tuple[int, int]] = None
    ) -> 'Data':
        """Specify validation for shape, column types and column constraints."""
        return cls(columns=columns, shape=shape)

    @classmethod
    def with_columns(cls, columns: Dict[str, ColumnSpec]) -> 'Data':
        """Specify expected columns and their types."""
        return cls(columns=columns)
    
    @classmethod
    def with_shape(cls, rows: int, cols: int) -> 'Data':
        """Specify the expected shape of the DataFrame (rows, columns)."""
        return cls(shape=(rows, cols))
    
    def validate_metadata(self) -> None:
        """One-time validation for instance metadata."""
        if not self.path:
            validation_collector.add_error(self.stage, "Data validation", "File path is required.")
        
        if self.shape:
            if not isinstance(self.shape, tuple) or len(self.shape) != 2:
                validation_collector.add_error(self.stage, "Data validation", "Shape must be a tuple of (rows, columns).")

    def validate_data(self) -> None:
        """Validate the columns and shape of the data."""
        data = DataIOHandler.read_data(self.path)
        
        if self.shape and data.shape != self.shape:
            validation_collector.add_error(self.stage, "Data validation", f"Expected DataFrame shape {self.shape}, got {data.shape}.")
        
        if self.columns:
            self.validate_columns(data)
    
    def validate_columns(self, data: DataFrame) -> None:
        """Validate columns based on specified types and constraints."""
        for col_name, col_spec in self.columns.items():
            if col_name not in data.columns:
                validation_collector.add_error(self.stage, "Data validation", f"Column '{col_name}' is missing from the data.")
            
            expected_type = col_spec.get('type')
            constraints = col_spec.get('constraints', [])
            
            if expected_type:
                polars_type = self.type_mapping.get(expected_type)  # TODO: better ways to handle this?
                if not polars_type:
                    validation_collector.add_error(self.stage, "Data validation", f"Unsupported type {expected_type}.")
                if data[col_name].dtype != polars_type:
                    validation_collector.add_error(self.stage, "Data validation", f"Column '{col_name}' must be of type {expected_type.__name__}.")
            
            for constraint in constraints:
                constraint(data[col_name])
