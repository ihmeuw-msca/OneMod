"""Unit tests for Jobmon-related functions."""

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
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("extension", ["json", "pkl", "toml", "yaml"])
def test_resources_path_to_dict(resource_dir, extension):
    file_path = resource_dir / f"resources.{extension}"
    assert isinstance(jb.get_resources(file_path), dict)
    assert isinstance(jb.get_resources(str(file_path)), dict)


@pytest.mark.unit
@pytest.mark.requires_jobmon
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


@pytest.mark.unit
@pytest.mark.requires_jobmon
@pytest.mark.parametrize(
    "submodel_args", [[], ["subsets"], ["paramsets"], ["subsets", "paramsets"]]
)
@pytest.mark.parametrize(
    "kwargs", [{}, {"key1": "dummy", "key2": {"key3": "dummy"}}]
)
def test_command_template(submodel_args, kwargs):
    expected_template = (
        "{entrypoint} --config {config} --method {method} --stages {stages}"
    )
    for key, value in kwargs.items():
        if isinstance(value, dict):
            expected_template += f" --{key} '{{{key}}}'"
        else:
            expected_template += f" --{key} {{{key}}}"
    for submodel_arg in submodel_args:
        expected_template += f" --{submodel_arg} '{{{submodel_arg}}}'"
    command_template = jb.get_command_template("run", submodel_args, **kwargs)
    assert command_template == expected_template


@pytest.mark.unit
@pytest.mark.requires_jobmon
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
@pytest.mark.requires_jobmon
def test_stage_task_resources_only():
    task_resources = jb.get_task_resources(
        TASK_RESOURCES, "cluster", "stage2", "method"
    )
    assert task_resources == {"resource1": "stage2_value1"}


@pytest.mark.unit
@pytest.mark.requires_jobmon
def test_method_task_resources_only():
    task_resources = jb.get_task_resources(
        TASK_RESOURCES, "cluster", "stage3", "method"
    )
    assert task_resources == {"resource1": "stage3_method_value1"}


@pytest.mark.unit
@pytest.mark.requires_jobmon
def test_no_task_resources():
    task_resources = jb.get_task_resources(
        TASK_RESOURCES, "cluster", "stage4", "method"
    )
    assert task_resources == {}


@pytest.mark.unit
@pytest.mark.requires_jobmon
def test_no_cluster_resources():
    task_resources = jb.get_task_resources(
        TASK_RESOURCES, "dummy", "stage1", "method"
    )
    assert task_resources == {}
