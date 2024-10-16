import os
from typing import Generator

from dotenv import load_dotenv
import pytest

from onemod.validation.error_handling import (
    validation_context,
    ValidationErrorCollector,
)

load_dotenv()


@pytest.fixture(scope="session")
def test_assets_dir():
    """Fixture to provide the test assets directory, as set in the environment variable."""
    test_dir = os.getenv("TEST_ASSETS_DIR")
    if not test_dir:
        raise EnvironmentError(
            "The TEST_ASSETS_DIR environment variable is not set."
        )
    return test_dir


@pytest.fixture
def validation_collector() -> Generator[ValidationErrorCollector, None, None]:
    """Fixture that manages the validation context for tests."""
    with validation_context() as collector:
        yield collector
