from pathlib import Path
from typing import Any, ClassVar

from polars import Boolean, DataFrame, Float64, Int64, String
from pydantic import BaseModel

from onemod.constraints import Constraint
from onemod.dtypes.column_spec import ColumnSpec
from onemod.fsutils.data_loader import DataLoader
from onemod.fsutils.io import configio_dict, dataio_dict
from onemod.validation.error_handling import (
    ValidationErrorCollector,
    handle_error,
)


class Data(BaseModel):
    stage: str | None = None
    methods: list[str] | None = None
    format: str | None = None
    path: Path | None = None
    shape: tuple[int, int] | None = None
    columns: dict[str, ColumnSpec] | None = None
    type_mapping: ClassVar[dict[type, Any]] = {
        bool: Boolean,
        int: Int64,
        float: Float64,
        str: String,
    }

    def model_post_init(self, *args, **kwargs) -> None:
        if self.format is None and self.path is None:
            raise ValueError("Data format and path cannot both be None")
        if self.path is not None:
            expected_format = self.path.suffix[1:] or "directory"
            if self.format is None:
                self.format = expected_format
            else:
                if self.format != expected_format:
                    raise ValueError("Data format and path do not match")

    def validate_metadata(
        self, kind: str, collector: ValidationErrorCollector | None = None
    ) -> None:
        """One-time validation for instance metadata."""
        if self.path is None:
            handle_error(
                self.stage,
                "Data validation",
                ValueError,
                "File path is required.",
                collector,
            )

        if self.format is not None:
            fextn = "." + self.format
            if fextn not in dataio_dict and fextn not in configio_dict:
                handle_error(
                    self.stage,
                    "Data validation",
                    ValueError,
                    f"Unsupported file format {self.format}.",
                    collector,
                )

        if self.shape:
            if not isinstance(self.shape, tuple) or len(self.shape) != 2:
                handle_error(
                    self.stage,
                    "Data validation",
                    ValueError,
                    "Shape must be a tuple of (rows, columns).",
                    collector,
                )

        if self.columns:
            for col_name, col_spec in self.columns.items():
                if (
                    hasattr(col_spec, "type")
                    and col_spec.type not in self.type_mapping
                ):
                    handle_error(
                        self.stage,
                        "Data validation",
                        ValueError,
                        f"Unsupported type {col_spec.type} for column {col_name}.",
                        collector,
                    )
                if hasattr(col_spec, "constraints") and col_spec.constraints:
                    for constraint in col_spec.constraints:
                        if not isinstance(constraint, Constraint):
                            handle_error(
                                self.stage,
                                "Data validation",
                                ValueError,
                                f"Invalid constraint specified for column {col_name}.",
                                collector,
                            )

    def validate_shape(
        self, data: DataFrame, collector: ValidationErrorCollector | None = None
    ) -> None:
        """Validate the shape of the data."""
        if data.shape != self.shape:
            handle_error(
                self.stage,
                "Data validation",
                ValueError,
                f"Expected DataFrame shape {self.shape}, got {data.shape}.",
                collector,
            )

    def validate_data(
        self,
        data: DataFrame | None,
        collector: ValidationErrorCollector | None = None,
    ) -> None:
        """Validate the columns and shape of the data."""
        if data is None and self.path is not None:
            try:
                data = DataLoader().load(self.path)
            except Exception as e:
                handle_error(
                    self.stage,
                    "Data validation",
                    e.__class__,
                    str(e),
                    collector,
                )
                return

        if self.shape:
            self.validate_shape(data, collector)

        if self.columns:
            self.validate_columns(data, collector)

    def validate_columns(
        self, data: DataFrame, collector: ValidationErrorCollector | None = None
    ) -> None:
        """Validate columns based on specified types and constraints."""
        if self.columns is None:
            return

        for col_name, col_spec in self.columns.items():
            if col_name not in data.columns:
                handle_error(
                    self.stage,
                    "Data validation",
                    ValueError,
                    f"Column '{col_name}' is missing from the data.",
                    collector,
                )

            expected_type = col_spec.type or None
            constraints = col_spec.constraints or []

            if expected_type:
                polars_type = self.type_mapping.get(
                    expected_type
                )  # TODO: better ways to handle this?
                if not polars_type:
                    handle_error(
                        self.stage,
                        "Data validation",
                        ValueError,
                        f"Unsupported type {expected_type}.",
                        collector,
                    )
                if data[col_name].dtype != polars_type:
                    handle_error(
                        self.stage,
                        "Data validation",
                        ValueError,
                        f"Column '{col_name}' must be of type {expected_type.__name__}.",
                        collector,
                    )

            for constraint in constraints:
                constraint.use_validation(data[col_name])
