"""Integration tests for Jobmon backend."""

import pytest

from onemod.backend import jobmon_backend as jb


@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_simple_submodel_args(simple_pipeline, method):
    stage = simple_pipeline.stages["run_1"]
    assert (
        jb.get_submodel_args(stage, method, subsets=None, paramsets=None) == {}
    )
    assert (
        jb.get_submodel_args(
            stage, method, subsets={"dummy_id": 1}, paramsets={"dummy_param": 1}
        )
        == {}
    )


@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize(
    "subsets",
    [
        None,
        {"sex_id": 1},
        {"sex_id": 1, "age_group_id": 1},
        {"sex_id": 1, "age_group_id": [1, 2]},
    ],
)
@pytest.mark.parametrize(
    "paramsets",
    [
        None,
        {"param1": 1},
        {"param1": 1, "param2": 1},
        {"param1": 1, "param2": [1, 2]},
    ],
)
def test_parallel_submodel_args(parallel_pipeline, method, subsets, paramsets):
    stage = parallel_pipeline.stages["run_1"]
    submodel_args = jb.get_submodel_args(
        stage, method, subsets=subsets, paramsets=paramsets
    )
    for arg_name, arg_dict in [["subsets", subsets], ["paramsets", paramsets]]:
        node_vals = getattr(stage, arg_name)
        if arg_dict is not None:
            node_vals = stage.get_subset(node_vals, arg_dict)
        assert submodel_args.get(arg_name) == [
            str(node_val) for node_val in node_vals.to_dict(orient="records")
        ]


@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_submodel_args_subsets_only(parallel_pipeline, method):
    stage = parallel_pipeline.stages["fit_2"]
    submodel_args = jb.get_submodel_args(
        stage, method, subsets=None, paramsets=None
    )
    assert submodel_args["subsets"] == [
        str(subset) for subset in stage.subsets.to_dict(orient="records")
    ]
    "paramsets" not in submodel_args


@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_submodel_args_paramsets_only(parallel_pipeline, method):
    stage = parallel_pipeline.stages["predict_3"]
    submodel_args = jb.get_submodel_args(
        stage, method, subsets=None, paramsets=None
    )
    "subsets" not in submodel_args
    assert submodel_args["paramsets"] == [
        str(subset) for subset in stage.paramsets.to_dict(orient="records")
    ]


@pytest.mark.integration
def test_parallel_submodel_args_collect(parallel_pipeline):
    stage = parallel_pipeline.stages["run_1"]
    assert (
        jb.get_submodel_args(stage, "collect", subsets=None, paramsets=None)
        == {}
    )
    assert (
        jb.get_submodel_args(
            stage, "collect", subsets={"sex_id": 1}, paramsets={"param": 1}
        )
        == {}
    )


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
        "stage", "method", tool, resources, submodel_args=[]
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


@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["fit_2", "predict_3"]])
def test_simple_upstream(simple_pipeline, method, stages):
    # TODO: Update once method-specific dependencies implemented
    task_dict = {}
    for stage_name in simple_pipeline.get_execution_order(stages):
        stage = simple_pipeline.stages[stage_name]
        if method not in stage.skip:
            upstream_tasks = jb.get_upstream_tasks(
                stage, method, simple_pipeline.stages, task_dict, stages=stages
            )

            if stage_name == "run_1" or stages is not None:
                assert upstream_tasks == []
            else:
                assert upstream_tasks == [f"run_1__{method}"]

            task_dict[stage_name] = [f"{stage_name}__{method}"]


@pytest.mark.integration
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["fit_2", "predict_3"]])
def test_parallel_upstream(parallel_pipeline, method, stages):
    task_dict = {}
    for stage_name in parallel_pipeline.get_execution_order(stages):
        stage = parallel_pipeline.stages[stage_name]
        upstream_tasks = jb.get_upstream_tasks(
            stage, method, parallel_pipeline.stages, task_dict, stages=stages
        )

        if stage_name == "run_1" or stages is not None:
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
            f"{stage_name}__{method}__submodel_{ii + 1}"
            for ii in range(len(stage.get_submodels()))
        ]

        if method in stage.collect_after:
            task_dict[stage_name].append(f"{stage_name}__collect")


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
