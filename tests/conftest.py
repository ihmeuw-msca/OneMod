from typing import Generator

import pytest

from onemod.validation.error_handling import validation_context, ValidationErrorCollector

@pytest.fixture
def validation_collector() -> Generator[ValidationErrorCollector, None, None]:
    """Fixture that manages the validation context for tests."""
    with validation_context() as collector:
        yield collector  # Provide the collector to the test
