from typing import Generator

import pandas as pd
import pytest

from onemod.fsutils.config_loader import ConfigLoader
from onemod.validation.error_handling import (
    ValidationErrorCollector,
    validation_context,
)
from tests.helpers.get_expected_args import get_expected_args
from tests.helpers.orchestration_helpers import (
    setup_parallel_pipeline,
    setup_simple_pipeline,
)


@pytest.fixture
def small_input_data(tmp_path_factory):
    """Create a dynamic test dataset based on expected stage inputs from get_expected_args()."""
    tmp_dir = tmp_path_factory.mktemp("data")
    parquet_path = tmp_dir / "small_data.parquet"

    expected = get_expected_args()
    dfs = []

    for stage_args in expected.values():
        subsets = stage_args.get("subsets")
        if subsets is not None:
            df = subsets.copy()
            # Add required columns not in the subset
            if "age_group_id" not in df.columns:
                df["age_group_id"] = 2
            if "location_id" not in df.columns:
                df["location_id"] = 6
            if "region_id" not in df.columns:
                df["region_id"] = 5.0
            if "super_region_id" not in df.columns:
                df["super_region_id"] = 4.0
            if "sex_id" not in df.columns:
                df["sex_id"] = 1
            if "year_id" not in df.columns:
                df["year_id"] = 1990

            # Add dummy values for required numeric columns
            df["fake_observation_column"] = 0.1
            df["fake_prediction_column"] = 0.2
            df["fake_weights_column"] = 1.0
            df["cov1"] = 0.5
            df["cov2"] = 1.5
            df["cov3"] = 2.5
            df["holdout1"] = 0
            df["holdout2"] = 1
            df["holdout3"] = 0
            df["adult_hiv_death_rate"] = 0.01

            dfs.append(df)

    # Add a few default rows for the preprocessing stage, which has no subset requirements
    dfs.append(
        pd.DataFrame(
            [
                {
                    "age_group_id": 1,
                    "location_id": 6,
                    "sex_id": 1,
                    "year_id": 1990,
                    "region_id": 5.0,
                    "super_region_id": 4.0,
                    "fake_observation_column": 0.1,
                    "fake_prediction_column": 0.2,
                    "fake_weights_column": 1.0,
                    "cov1": 0.5,
                    "cov2": 1.5,
                    "cov3": 2.5,
                    "holdout1": 0,
                    "holdout2": 1,
                    "holdout3": 0,
                    "adult_hiv_death_rate": 0.01,
                }
            ]
        )
    )

    df_final = pd.concat(dfs, ignore_index=True).drop_duplicates()
    df_final.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture
def test_base_dir(tmp_path_factory):
    test_base_dir = tmp_path_factory.mktemp("test_base_dir")
    return test_base_dir


@pytest.fixture(scope="session")
def validation_collector() -> Generator[ValidationErrorCollector, None, None]:
    """Fixture that manages the validation context for tests."""
    with validation_context() as collector:
        yield collector


@pytest.fixture(scope="function")
def simple_pipeline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("jobmon_test_dir")
    pipeline = setup_simple_pipeline(directory)
    return pipeline


@pytest.fixture(scope="function")
def second_simple_pipeline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("second_jobmon_test_dir")
    pipeline = setup_simple_pipeline(directory)
    return pipeline


@pytest.fixture(scope="function")
def parallel_pipeline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("jobmon_test_dir")
    pipeline = setup_parallel_pipeline(directory)
    return pipeline


@pytest.fixture(scope="function")
def second_parallel_pipeline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("second_jobmon_test_dir")
    pipeline = setup_parallel_pipeline(directory)
    return pipeline


@pytest.fixture(scope="module")
def resource_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("resources_test_dir")
    config_loader = ConfigLoader()
    for extension in ["json", "pkl", "toml", "yaml"]:
        config_loader.dump(
            {"tool_resources": {"dummy": {"queue": "null.q"}}},
            directory / f"resources.{extension}",
        )
    return directory


@pytest.fixture(scope="function")
def jobmon_dummy_cluster_env(monkeypatch):
    monkeypatch.setenv("JOBMON__DISTRIBUTOR__POLL_INTERVAL", "1")
    monkeypatch.setenv("JOBMON__HEARTBEAT__WORKFLOW_RUN_INTERVAL", "1")
    monkeypatch.setenv("JOBMON__HEARTBEAT__TASK_INSTANCE_INTERVAL", "1")
