from contextlib import contextmanager
from typing import Any, Generator

from pydantic import BaseModel


class ValidationErrorReport(BaseModel):
    stage: str
    error_type: str
    message: str
    details: dict[str, Any] | None = None


class ValidationErrorCollector(BaseModel):
    errors: list[ValidationErrorReport] = []

    def add_error(self, stage: str, error_type: str, message: str, details: dict = None) -> None:
        error_report = ValidationErrorReport(stage=stage, error_type=error_type, message=message, details=details)
        self.errors.append(error_report)

    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def get_errors(self) -> list[ValidationErrorReport]:
        return self.errors

collector = None

@contextmanager
def validation_context() -> Generator[ValidationErrorCollector, None, None]:
    """Context manager for managing validation error collection."""
    global collector
    collector = ValidationErrorCollector()
    try:
        yield collector
    finally:
        collector = None  # Clean up after validation
