from contextlib import contextmanager
from typing import Any, Generator

from pydantic import BaseModel


class ValidationErrorReport(BaseModel):
    stage: str
    error_category: str
    message: str
    details: dict[str, Any] | None = None


class ValidationErrorCollector(BaseModel):
    errors: list[ValidationErrorReport] = []

    def add_error(self, stage: str, error_category: str, message: str, details: dict = None) -> None:
        error_report = ValidationErrorReport(stage=stage, error_category=error_category, message=message, details=details)
        self.errors.append(error_report)

    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def get_errors(self) -> list[ValidationErrorReport]:
        return self.errors

validation_collector = None

@contextmanager
def validation_context() -> Generator[ValidationErrorCollector, None, None]:
    """Context manager for managing validation error collection."""
    global validation_collector
    validation_collector = ValidationErrorCollector()
    try:
        yield validation_collector
    finally:
        validation_collector = None  # Clean up after validation

def handle_error(
    stage: str,
    error_category: str,
    error_type: type[Exception],
    message: str,
    collector: ValidationErrorCollector | None = None,
    details: dict | None = None
) -> None:
    """Handle an error by either raising it or adding it to the validation collector."""
    if collector:
        collector.add_error(stage, error_category, message, details)
    else:
        raise error_type(message)
