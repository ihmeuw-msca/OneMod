"""Integration tests for Jobmon backend."""

import pytest

from onemod.backend import jobmon_backend as jb


@pytest.mark.integration
def test_config_path(simple_pipeline):
    config_path = str(
        simple_pipeline.directory / f"{simple_pipeline.name}.json"
    )
    for stage in simple_pipeline.stages.values():
        assert jb.get_config_path(stage) == config_path


@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_simple_submodel_args(simple_pipeline, method):
    for stage in simple_pipeline.stages.values():
        submodel_args = jb.get_submodel_args(
            stage, method, subsets={}, paramsets={}
        )
        assert submodel_args == {}


# TODO: Add tests for subsets/paramsets
@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict", "collect"])
def test_parallel_submodel_args(parallel_pipeline, method):
    for stage in parallel_pipeline.stages.values():
        submodel_args = jb.get_submodel_args(
            stage, method, subsets={}, paramsets={}
        )
        if method == "collect":
            assert submodel_args == {}
        else:
            for attr, submodel_arg in [
                ["subsets", "subset"],
                ["paramsets", "paramset"],
            ]:
                expected = getattr(stage, attr)
                observed = submodel_args.get(submodel_arg)
                if expected is not None:
                    assert observed == [
                        str(node_val)
                        for node_val in expected.to_dict(orient="records")
                    ]
                else:
                    assert observed is None


# TODO: Add tests for method_args, submodel_args
@pytest.mark.integration
@pytest.mark.parametrize("stage_cluster", ["cluster", "dummy"])
def test_task_template(stage_cluster):
    resources = {
        "tool_resources": {"cluster": {"queue": "tool.q"}},
        "task_template_resources": {
            "stage": {stage_cluster: {"queue": "stage.q"}}
        },
    }
    tool = jb.get_tool("pipeline", "method", "cluster", resources)
    task_template = jb.get_task_template(
        "stage", "method", tool, resources, method_args=[], submodel_args=[]
    )
    default_cluster = task_template.default_cluster_name
    default_resources = task_template.default_compute_resources_set
    assert task_template.template_name == "stage_method"
    if stage_cluster == "cluster":
        assert default_cluster == "cluster"
        assert default_resources == {stage_cluster: {"queue": "stage.q"}}
    else:
        assert default_cluster == ""
        assert default_resources == {}


# TODO: Add tests for stages
@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_simple_upstream(simple_pipeline, method):
    # TODO: Update once method-specific dependencies implemented
    task_dict = {}
    for stage_name in simple_pipeline.get_execution_order():
        stage = simple_pipeline.stages[stage_name]
        if method not in stage.skip:
            upstream_tasks = jb.get_upstream_tasks(
                stage, method, simple_pipeline.stages, task_dict, stages=None
            )

            if stage_name == "run_1":
                assert upstream_tasks == []
            elif stage_name == "fit_2":
                assert upstream_tasks == [f"run_1__{method}"]
            elif stage_name == "predict_3":
                assert upstream_tasks == [f"run_1__{method}"]

            task_dict[stage_name] = [f"{stage_name}__{method}"]


# TODO: Add tests for stages
@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_upstream(parallel_pipeline, method):
    task_dict = {}
    for stage_name in parallel_pipeline.get_execution_order():
        stage = parallel_pipeline.stages[stage_name]
        upstream_tasks = jb.get_upstream_tasks(
            stage, method, parallel_pipeline.stages, task_dict, stages=None
        )

        if stage_name == "run_1":
            assert upstream_tasks == []
        elif stage_name == "fit_2":
            assert upstream_tasks == ["run_1__collect"]
        elif stage_name == "predict_3":
            assert upstream_tasks == ["run_1__collect"]
        elif stage_name == "run_4":
            if method == "run":
                assert upstream_tasks == [
                    "fit_2__collect",
                    "predict_3__collect",
                ]
            elif method == "fit":
                assert upstream_tasks == [
                    "fit_2__collect",
                    "predict_3__fit__submodel_1",
                    "predict_3__fit__submodel_2",
                ]
            elif method == "predict":
                assert upstream_tasks == [
                    "fit_2__predict__submodel_1",
                    "fit_2__predict__submodel_2",
                    "predict_3__collect",
                ]

        task_dict[stage_name] = [
            f"{stage_name}__{method}__submodel_1",
            f"{stage_name}__{method}__submodel_2",
        ]

        if method in stage.collect_after:
            task_dict[stage_name].append(f"{stage_name}__collect")


@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_subset_simple_upstream(simple_pipeline, method):
    # Make sure upstream stage not added if not in subset
    pass


@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_subset_parallel_upstream(parallel_pipeline, method):
    # Make sure upstream stage not added if not in subset
    pass


def test_cant_evaluate_collect():
    # Make sure error raised if trying to call collect method with jobmon
    pass


def test_missing_input():
    # Make sure error raised if upstream input doesnt exist
    # Used for paths or upstream output created by stages not in subset
    pass


def test_pipeline_tasks():
    # Correct name (name)
    # Correct cluster (cluster_name)
    # Correct resources (compute_resources)
    # Correct dependencies (upstream_tasks, downstream_tasks)
    # Correct args (node["node_args"], task_args, op_args)
    # Correct command template (command)
    # Correct use of skip attribute
    # Correct use of collect_after
    # With and without stage subset
    pass


def test_stage_tasks():
    # Correct name (name)
    # Correct cluster (cluster_name)
    # Correct resources (compute_resources)
    # Correct dependencies (upstream_tasks, downstream_tasks)
    # Correct args (node["node_args"], task_args, op_args)
    # Correct command template (command)
    # Correct use of collect_after
    # With and without stage subset
    pass
