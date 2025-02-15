from contextlib import contextmanager
from typing import Any, Generator

from pydantic import BaseModel


class ValidationErrorReport(BaseModel):
    stage: str | None
    error_category: str
    message: str
    details: dict[str, Any] | None = None


class ValidationErrorCollector(BaseModel):
    errors: list[ValidationErrorReport] = []

    def add_error(
        self,
        stage: str | None,
        error_category: str,
        message: str,
        details: dict | None = None,
    ) -> None:
        error_report = ValidationErrorReport(
            stage=stage,
            error_category=error_category,
            message=message,
            details=details,
        )
        self.errors.append(error_report)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def get_errors(self) -> list[ValidationErrorReport]:
        return self.errors

    def raise_errors(self) -> None:
        if self.has_errors():
            raise ValidationException(self.errors)


class ValidationException(Exception):
    def __init__(self, errors: list[ValidationErrorReport]):
        self.errors = errors
        super().__init__(self._format_errors())

    def _format_errors(self) -> str:
        messages = [
            f"Stage '{error.stage}' - {error.error_category}: {error.message}"
            for error in self.errors
        ]
        return "\n".join(messages)


@contextmanager
def validation_context() -> Generator[ValidationErrorCollector, None, None]:
    """Context manager for managing validation error collection."""
    validation_collector = ValidationErrorCollector()
    try:
        yield validation_collector
    finally:
        pass


def handle_error(
    stage: str | None,
    error_category: str,
    error_type: type[Exception],
    message: str,
    collector: ValidationErrorCollector | None = None,
    details: dict | None = None,
) -> None:
    """Handle an error by either raising it or adding it to the validation collector."""
    if collector:
        collector.add_error(stage, error_category, message, details)
    else:
        raise error_type(message)
