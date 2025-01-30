"""Unit tests for Jobmon-related functions."""
# TODO: Test resource errors if missing required cluster resources

import sys
from pathlib import Path

import pytest

from onemod.backend import jobmon_backend as jb

TASK_RESOURCES = {
    "task_template_resources": {
        "stage1": {
            "cluster": {
                "resource1": "stage1_value1",
                "resource2": "stage1_value2",
            }
        },
        "stage2": {"cluster": {"resource1": "stage2_value1"}},
        "stage1_method": {
            "cluster": {
                "resource1": "stage1_method_value1",
                "resource3": "stage1_method_value3",
            }
        },
        "stage3_method": {"cluster": {"resource1": "stage3_method_value1"}},
    }
}


@pytest.mark.unit
@pytest.mark.parametrize("extension", ["json", "pkl", "toml", "yaml"])
def test_resources_path_to_dict(resource_dir, extension):
    file_path = resource_dir / f"resources.{extension}"
    assert isinstance(jb.get_resources(file_path), dict)
    assert isinstance(jb.get_resources(str(file_path)), dict)


@pytest.mark.unit
def get_entrypoint():
    python_path = "/path/to/python/env"
    entrypoint = "/path/to/python/onemod"
    assert jb.get_entrypoint(Path(python_path)) == entrypoint
    assert jb.get_entrypoint(python_path) == entrypoint
    assert jb.get_entrypoint() == str(Path(sys.executable).parent / "onemod")


@pytest.mark.unit
def test_stage_and_method_task_resources():
    task_resources = jb.get_task_resources(
        TASK_RESOURCES, "cluster", "stage1", "method"
    )
    expected_resources = {
        "resource1": "stage1_method_value1",
        "resource2": "stage1_value2",
        "resource3": "stage1_method_value3",
    }
    assert task_resources == expected_resources


@pytest.mark.unit
def test_stage_task_resources_only():
    task_resources = jb.get_task_resources(
        TASK_RESOURCES, "cluster", "stage2", "method"
    )
    assert task_resources == {"resource1": "stage2_value1"}


@pytest.mark.unit
def test_method_task_resources_only():
    task_resources = jb.get_task_resources(
        TASK_RESOURCES, "cluster", "stage3", "method"
    )
    assert task_resources == {"resource1": "stage3_method_value1"}


@pytest.mark.unit
def test_no_task_resources():
    task_resources = jb.get_task_resources(
        TASK_RESOURCES, "cluster", "stage4", "method"
    )
    assert task_resources == {}


@pytest.mark.unit
def test_no_cluster_resources():
    task_resources = jb.get_task_resources(
        TASK_RESOURCES, "dummy", "stage1", "method"
    )
    assert task_resources == {}


# TODO: Add tests for method_args
@pytest.mark.unit
def test_command_template():
    submodel_args = []
    expected_template = (
        "{entrypoint} --config {config}"
        " --method dummy_method --stages dummy_stage"
    )
    for submodel_arg in ["", "subset", "paramset"]:
        if submodel_arg:
            submodel_args.append(submodel_arg)
            expected_template += f" --{submodel_arg} '{{{submodel_arg}}}'"
        command_template = jb.get_command_template(
            "dummy_stage",
            "dummy_method",
            method_args=[],
            submodel_args=submodel_args,
        )
        assert command_template == expected_template


@pytest.mark.unit
def test_tool():
    tool = jb.get_tool(
        "pipeline",
        "method",
        "cluster",
        {"tool_resources": {"cluster": {"queue": "null.q"}}},
    )
    assert tool.name == "pipeline_method"
    assert tool.default_cluster_name == "cluster"
    assert tool.default_compute_resources_set == {
        "cluster": {"queue": "null.q"}
    }
