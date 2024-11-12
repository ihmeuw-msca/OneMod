import os
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv

from onemod.validation.error_handling import (
    ValidationErrorCollector,
    validation_context,
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
def small_input_data(request, test_assets_dir):
    """Fixture providing path to test input data for tests marked with requires_data."""
    if request.node.get_closest_marker("requires_data") is None:
        pytest.skip("Skipping test because it requires data assets.")

    small_input_data_path = Path(
        test_assets_dir, "e2e", "example1", "data", "small_data.parquet"
    )
    return small_input_data_path


@pytest.fixture
def dummy_resources(request, test_assets_dir):
    """Fixture providing path to test resources for tests marked with requires_data."""
    if request.node.get_closest_marker("requires_data") is None:
        pytest.skip("Skipping test because it requires data assets.")

    dummy_resources_path = Path(
        test_assets_dir, "e2e", "example1", "config", "jobmon", "resources.yaml"
    )
    return dummy_resources_path


@pytest.fixture
def test_base_dir(tmp_path_factory):
    test_base_dir = tmp_path_factory.mktemp("test_base_dir")
    return test_base_dir


@pytest.fixture(scope="session")
def validation_collector() -> Generator[ValidationErrorCollector, None, None]:
    """Fixture that manages the validation context for tests."""
    with validation_context() as collector:
        yield collector
