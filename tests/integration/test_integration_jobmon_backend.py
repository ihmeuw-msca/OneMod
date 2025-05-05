"""Integration tests for Jobmon backend."""

from collections import defaultdict
from pathlib import Path
from unittest import mock

import pytest

try:
    from onemod.backend import jobmon_backend as jb
except ImportError:
    pass


@pytest.mark.integration
@pytest.mark.requires_jobmon
def test_simple_submodel_args(simple_pipeline):
    stage = simple_pipeline.stages["run_1"]
    method = "run"
    assert (
        jb.get_submodel_args(stage, method, subsets=None, paramsets=None) == {}
    )
    assert (
        jb.get_submodel_args(
            stage, method, subsets={"sex_id": 1}, paramsets={"param": 1}
        )
        == {}
    )


@pytest.mark.integration
@pytest.mark.requires_jobmon
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
def test_parallel_submodel_args(parallel_pipeline, subsets, paramsets):
    stage = parallel_pipeline.stages["run_1"]
    submodel_args = jb.get_submodel_args(
        stage, "run", subsets=subsets, paramsets=paramsets
    )
    for arg_name, arg_dict in [["subsets", subsets], ["paramsets", paramsets]]:
        node_vals = getattr(stage, arg_name)
        if arg_dict is not None:
            node_vals = stage.get_subset(node_vals, arg_dict)
        assert submodel_args.get(arg_name) == [
            str(node_val) for node_val in node_vals.to_dict(orient="records")
        ]


@pytest.mark.integration
@pytest.mark.requires_jobmon
def test_parallel_submodel_args_subsets_only(parallel_pipeline):
    stage = parallel_pipeline.stages["fit_2"]
    submodel_args = jb.get_submodel_args(
        stage, "run", subsets=None, paramsets=None
    )
    assert submodel_args["subsets"] == [
        str(subset) for subset in stage.subsets.to_dict(orient="records")
    ]
    "paramsets" not in submodel_args


@pytest.mark.integration
@pytest.mark.requires_jobmon
def test_parallel_submodel_args_paramsets_only(parallel_pipeline):
    stage = parallel_pipeline.stages["predict_3"]
    submodel_args = jb.get_submodel_args(
        stage, "run", subsets=None, paramsets=None
    )
    "subsets" not in submodel_args
    assert submodel_args["paramsets"] == [
        str(subset) for subset in stage.paramsets.to_dict(orient="records")
    ]


@pytest.mark.integration
@pytest.mark.requires_jobmon
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
@pytest.mark.requires_jobmon
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
        "stage",
        "method",
        tool,
        resources,
        submodel_args=[],
        template_prefix=None,
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
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["fit_2", "predict_3"]])
def test_simple_upstream(simple_pipeline, method, stages):
    # TODO: Update once method-specific dependencies implemented
    task_dict = {}
    for stage_name in simple_pipeline.get_execution_order(stages):
        stage = simple_pipeline.stages[stage_name]
        if method not in stage.skip:
            upstream_tasks = jb.get_upstream_tasks(
                stage=stage,
                method=method,
                stage_dict=simple_pipeline.stages,
                task_dict=task_dict,
                stages=stages,
                task_prefix=None,
                template_prefix=None,
            )

            if stage_name == "run_1" or stages is not None:
                assert upstream_tasks == []
            else:
                assert upstream_tasks == [f"run_1__{method}"]

            task_dict[stage_name] = [f"{stage_name}__{method}"]


@pytest.mark.integration
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["fit_2", "predict_3"]])
def test_parallel_upstream(parallel_pipeline, method, stages):
    task_dict = {}
    for stage_name in parallel_pipeline.get_execution_order(stages):
        stage = parallel_pipeline.stages[stage_name]
        upstream_tasks = jb.get_upstream_tasks(
            stage=stage,
            method=method,
            stage_dict=parallel_pipeline.stages,
            task_dict=task_dict,
            stages=stages,
            task_prefix=None,
            template_prefix=None,
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


@pytest.mark.integration
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["run_1", "fit_2"]])
def test_simple_pipeline_tasks(simple_pipeline, method, stages):
    cluster = "cluster"
    resources = {"tool_resources": {cluster: {"queue": "null.q"}}}
    python = "/path/to/python/env/bin/python"
    tasks = jb.get_pipeline_tasks(
        simple_pipeline,
        method,
        jb.get_tool(simple_pipeline.name, method, cluster, resources),
        resources=resources,
        python=python,
        stages=stages,
        external_upstream_tasks=None,
        task_prefix=None,
        template_prefix=None,
        max_attempts=1,
    )
    stages = list(simple_pipeline.stages.keys()) if stages is None else stages
    task_dict = {task.task_args["stages"]: task for task in tasks}
    for stage in simple_pipeline.stages.values():
        if stage.name in stages and method not in stage.skip:
            stage_task = task_dict[stage.name]
            upstream_stages = [
                task.task_args["stages"] for task in stage_task.upstream_tasks
            ]
            for upstream_stage in stage.dependencies:
                # assumes upstream_stage in stages and
                # method not in upstream_stage.skip
                assert upstream_stage in upstream_stages
        else:
            assert stage.name not in task_dict


@pytest.mark.integration
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["run_1", "fit_2", "predict_3"]])
def test_parallel_pipeline_tasks(parallel_pipeline, method, stages):
    cluster = "cluster"
    resources = {"tool_resources": {cluster: {"queue": "null.q"}}}
    python = "/path/to/python/env/bin/python"
    tasks = jb.get_pipeline_tasks(
        parallel_pipeline,
        method,
        jb.get_tool(parallel_pipeline.name, method, cluster, resources),
        resources=resources,
        python=python,
        stages=stages,
        external_upstream_tasks=None,
        task_prefix=None,
        template_prefix=None,
        max_attempts=1,
    )
    stages = list(parallel_pipeline.stages.keys()) if stages is None else stages
    task_dict = {task.task_args["stages"]: defaultdict(list) for task in tasks}
    for task in tasks:
        task_dict[task.task_args["stages"]][task.task_args["method"]].append(
            task
        )

    for stage in parallel_pipeline.stages.values():
        if stage.name in stages:
            method_tasks = task_dict[stage.name][method]
            collect_tasks = task_dict[stage.name]["collect"]
            assert len(method_tasks) == len(stage.get_submodels())

            if method in stage.collect_after:
                assert len(collect_tasks) == 1
                assert collect_tasks[0].upstream_tasks == set(method_tasks)
            else:
                assert len(collect_tasks) == 0

            for task in method_tasks:
                upstream_dict = {
                    upstream_task.task_args["stages"]: defaultdict(list)
                    for upstream_task in task.upstream_tasks
                }
                for upstream_task in task.upstream_tasks:
                    upstream_dict[upstream_task.task_args["stages"]][
                        upstream_task.task_args["method"]
                    ].append(upstream_task)
                for upstream_name in stage.dependencies:
                    # assumes upstream_stage in stages
                    upstream_stage = parallel_pipeline.stages[upstream_name]
                    if method in upstream_stage.collect_after:
                        assert len(upstream_dict[upstream_name][method]) == 0
                        assert len(upstream_dict[upstream_name]["collect"]) == 1
                    else:
                        assert len(upstream_dict[upstream_name][method]) == len(
                            upstream_stage.get_submodels()
                        )
                        assert len(upstream_dict[upstream_name]["collect"]) == 0
        else:
            assert stage.name not in task_dict


@pytest.mark.integration
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["run_1", "fit_2", "predict_3"]])
def test_parallel_pipeline_tasks_jobmon_args(parallel_pipeline, method, stages):
    cluster = "cluster"
    resources = {"tool_resources": {cluster: {"queue": "null.q"}}}
    python = "/path/to/python/env/bin/python"
    external_upstream_tasks = [
        jb.Task(
            node=mock.MagicMock(),
            task_args={"fake_arg_1": "fake_value"},
            op_args={"fake_arg_2": "fake_value"},
            name="fake_task",
            task_attributes=[],
        )
    ]
    task_prefix = "me_1234"
    template_prefix = "testing"
    max_attempts = 3
    tasks = jb.get_pipeline_tasks(
        parallel_pipeline,
        method,
        jb.get_tool(parallel_pipeline.name, method, cluster, resources),
        resources=resources,
        python=python,
        stages=stages,
        external_upstream_tasks=external_upstream_tasks,
        task_prefix=task_prefix,
        template_prefix=template_prefix,
        max_attempts=max_attempts,
    )
    stages = list(parallel_pipeline.stages.keys()) if stages is None else stages
    task_dict = {task.task_args["stages"]: defaultdict(list) for task in tasks}
    for task in tasks:
        task_dict[task.task_args["stages"]][task.task_args["method"]].append(
            task
        )

    for stage in parallel_pipeline.stages.values():
        if stage.name in stages:
            method_tasks = task_dict[stage.name][method]
            collect_tasks = task_dict[stage.name]["collect"]
            assert len(method_tasks) == len(stage.get_submodels())

            if method in stage.collect_after:
                assert len(collect_tasks) == 1
                assert collect_tasks[0].upstream_tasks == set(method_tasks)
            else:
                assert len(collect_tasks) == 0

            for task in method_tasks:
                stage_upstreams = [
                    upstream_task
                    for upstream_task in task.upstream_tasks
                    if "stages" in upstream_task.task_args
                ]
                external_upstreams = [
                    upstream_task
                    for upstream_task in task.upstream_tasks
                    if "stages" not in upstream_task.task_args
                ]
                upstream_dict = {
                    upstream_task.task_args["stages"]: defaultdict(list)
                    for upstream_task in stage_upstreams
                }
                for upstream_task in stage_upstreams:
                    upstream_dict[upstream_task.task_args["stages"]][
                        upstream_task.task_args["method"]
                    ].append(upstream_task)
                for upstream_name in stage.dependencies:
                    # assumes upstream_stage in stages
                    upstream_stage = parallel_pipeline.stages[upstream_name]
                    if method in upstream_stage.collect_after:
                        assert len(upstream_dict[upstream_name][method]) == 0
                        assert len(upstream_dict[upstream_name]["collect"]) == 1
                    else:
                        assert len(upstream_dict[upstream_name][method]) == len(
                            upstream_stage.get_submodels()
                        )
                        assert len(upstream_dict[upstream_name]["collect"]) == 0
                if external_upstreams:
                    assert external_upstreams == external_upstream_tasks
        else:
            assert stage.name not in task_dict


@pytest.mark.integration
@pytest.mark.requires_jobmon
def test_stage_tasks_basic(simple_pipeline):
    stage = simple_pipeline.stages["run_1"]
    method = "run"
    cluster = "cluster"
    resources = {"tool_resources": {cluster: {"queue": "null.q"}}}
    python = "/path/to/python/env/bin/python"
    entrypoint = str(Path(python).parent / "onemod")
    config = str(stage.dataif.get_path("config"))
    tasks = jb.get_stage_tasks(
        stage,
        method,
        jb.get_tool(simple_pipeline.name, method, cluster, resources),
        resources=resources,
        python=python,
        task_prefix=None,
        template_prefix=None,
        max_attempts=1,
    )
    task = tasks[0]

    assert len(tasks) == 1
    assert task.name == f"{stage.name}_{method}"
    assert task.cluster_name == ""
    assert task.compute_resources == {}
    assert task.command == jb.get_command_template(method, []).format(
        entrypoint=entrypoint, config=config, method=method, stages=stage.name
    )
    assert task.op_args == {"entrypoint": entrypoint}
    assert task.task_args == {"method": method, "stages": stage.name}
    assert task.node.node_args == {"config": config}


@pytest.mark.integration
@pytest.mark.requires_jobmon
@pytest.mark.parametrize(
    "kwargs", [{}, {"key1": "dummy", "key2": {"key3": "dummy"}}]
)
def test_stage_tasks_kwargs(simple_pipeline, kwargs):
    stage = simple_pipeline.stages["run_1"]
    method = "run"
    cluster = "cluster"
    resources = {"tool_resources": {cluster: {"queue": "null.q"}}}
    python = "/path/to/python/env/bin/python"
    entrypoint = str(Path(python).parent / "onemod")
    config = str(stage.dataif.get_path("config"))
    task = jb.get_stage_tasks(
        stage,
        method,
        jb.get_tool(simple_pipeline.name, method, cluster, resources),
        resources=resources,
        python=python,
        task_prefix=None,
        template_prefix=None,
        max_attempts=1,
        **kwargs,
    )[0]
    assert task.command == jb.get_command_template(method, [], **kwargs).format(
        entrypoint=entrypoint,
        config=config,
        method=method,
        stages=stage.name,
        **kwargs,
    )
    assert task.op_args == {"entrypoint": entrypoint}
    assert task.task_args == {
        **{"method": method, "stages": stage.name},
        **kwargs,
    }
    assert task.node.node_args == {"config": config}


@pytest.mark.integration
@pytest.mark.requires_jobmon
@pytest.mark.parametrize(
    "submodel",
    [
        [None, None],
        [{"sex_id": 1, "age_group_id": 1}, {"param1": 1, "param2": 1}],
    ],
)
@pytest.mark.parametrize("collect", [True, False])
def test_stage_tasks_submodels(parallel_pipeline, submodel, collect):
    subsets, paramsets = submodel
    stage = parallel_pipeline.stages["run_1"]
    method = "run"
    cluster = "cluster"
    resources = {"tool_resources": {cluster: {"queue": "null.q"}}}
    tasks = jb.get_stage_tasks(
        stage,
        method,
        jb.get_tool(parallel_pipeline.name, method, cluster, resources),
        resources=resources,
        python="/path/to/python/env/bin/python",
        subsets=subsets,
        paramsets=paramsets,
        collect=collect,
        task_prefix=None,
        template_prefix=None,
        max_attempts=1,
    )
    submodels = [
        [str(submodel[0]), str(submodel[1])]
        for submodel in stage.get_submodels(subsets, paramsets)
    ]
    if subsets is None:
        assert len(submodels) == len(stage.subsets) * len(stage.paramsets)
    else:
        assert len(submodels) == 1
    for task in tasks:
        if task.task_args["method"] != "collect":
            node_subsets = task.node.node_args["subsets"]
            node_paramsets = task.node.node_args["paramsets"]
            command_subsets = task.command.split("--")[4][9:-2]
            command_paramsets = task.command.split("--")[5][11:-1]
            assert node_subsets == command_subsets
            assert node_paramsets == command_paramsets
            assert [node_subsets, node_paramsets] in submodels
    if collect:
        assert len(tasks) == len(submodels) + 1
        assert tasks[-1].task_args["method"] == "collect"
    else:
        assert len(tasks) == len(submodels)


@pytest.mark.integration
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_stage_tasks_collect_after(parallel_pipeline, method):
    stage = parallel_pipeline.stages["fit_2"]
    cluster = "cluster"
    resources = {"tool_resources": {cluster: {"queue": "null.q"}}}
    tasks = jb.get_stage_tasks(
        stage,
        method,
        jb.get_tool(parallel_pipeline.name, method, cluster, resources),
        resources=resources,
        python="/path/to/python/env/bin/python",
        subsets={"sex_id": 1},
        paramsets={"param": 1},
        collect=True,
        task_prefix=None,
        template_prefix=None,
        max_attempts=1,
    )
    assert tasks[0].task_args["method"] == method
    if method == "predict":
        assert len(tasks) == 1
    else:
        assert len(tasks) == 2
        assert tasks[1].task_args["method"] == "collect"


@pytest.mark.integration
@pytest.mark.requires_jobmon
def test_stage_tasks_jobmon_args(simple_pipeline):
    stage = simple_pipeline.stages["run_1"]
    method = "run"
    cluster = "cluster"
    resources = {"tool_resources": {cluster: {"queue": "null.q"}}}
    python = "/path/to/python/env/bin/python"
    task_prefix = "me_1234"
    template_prefix = "testing"
    max_attempts = 3
    entrypoint = str(Path(python).parent / "onemod")
    config = str(stage.dataif.get_path("config"))
    tasks = jb.get_stage_tasks(
        stage,
        method,
        jb.get_tool(simple_pipeline.name, method, cluster, resources),
        resources=resources,
        python=python,
        task_prefix=task_prefix,
        template_prefix=template_prefix,
        max_attempts=max_attempts,
    )
    task = tasks[0]

    assert len(tasks) == 1
    assert task.name == f"{task_prefix}_{stage.name}_{method}"
    assert task.cluster_name == ""
    assert task.compute_resources == {}
    assert task.command == jb.get_command_template(method, []).format(
        entrypoint=entrypoint, config=config, method=method, stages=stage.name
    )
    assert task.max_attempts == max_attempts
    assert task.op_args == {"entrypoint": entrypoint}
    assert task.task_args == {"method": method, "stages": stage.name}
    assert task.node.node_args == {"config": config}
