from pathlib import Path
from typing import Any, ClassVar, Dict

from polars import Boolean, DataFrame, Int64, Float64, String
from pydantic import BaseModel, field_serializer

from onemod.constraints import Constraint
from onemod.dtypes.column_spec import ColumnSpec
from onemod.dtypes.filepath import FilePath
from onemod.utils import DataIOHandler
from onemod.validation.error_handling import (
    ValidationErrorCollector,
    handle_error,
)


class Data(BaseModel):
    stage: str
    path: Path | FilePath
    format: str = "parquet"
    shape: tuple[int, int] | None = None
    columns: Dict[str, ColumnSpec] | None = None
    type_mapping: ClassVar[Dict[type, Any]] = {
        bool: Boolean,
        int: Int64,
        float: Float64,
        str: String,
    }

    @field_serializer("path")
    def serialize_path(self, path, info):
        return str(path) if path else None

    def validate_metadata(
        self, kind: str, collector: ValidationErrorCollector | None = None
    ) -> None:
        """One-time validation for instance metadata."""
        if not self.path:
            handle_error(
                self.stage,
                "Data validation",
                ValueError,
                "File path is required.",
                collector,
            )
        else:
            # FIXME: Path won't exist until stage has been run
            # if kind == "input" and not self.path.exists():
            #     handle_error(
            #         self.stage,
            #         "Data validation",
            #         FileNotFoundError,
            #         f"File {self.path} does not exist.",
            #         collector,
            #     )
            if self.format not in DataIOHandler.supported_formats:
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
                    "type" in col_spec
                    and col_spec["type"] not in self.type_mapping
                ):
                    handle_error(
                        self.stage,
                        "Data validation",
                        ValueError,
                        f"Unsupported type {col_spec['type']} for column {col_name}.",
                        collector,
                    )
                if "constraints" in col_spec:
                    for constraint in col_spec["constraints"]:
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
        if data is None:
            try:
                data = DataIOHandler.read_data(self.path)
            except Exception as e:
                handle_error(
                    self.stage,
                    "Data validation",
                    e.__class__,
                    str(e),
                    collector,
                )

        if self.shape:
            self.validate_shape(data, collector)

        if self.columns:
            self.validate_columns(data, collector)

    def validate_columns(
        self, data: DataFrame, collector: ValidationErrorCollector | None = None
    ) -> None:
        """Validate columns based on specified types and constraints."""
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
                constraint.validate(data[col_name])
