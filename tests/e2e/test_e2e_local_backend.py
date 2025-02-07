"""End-to-end tests for local backend."""

import pytest
import tests.helpers.orchestration_helpers as helpers

KWARGS = {
    "backend": "local",
    "cluster": None,
    "resources": None,
    "python": None,
}

STAGE_KWARGS = {**KWARGS, "subsets": None, "paramsets": None, "collect": None}


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_simple_pipeline_method(simple_pipeline, method):
    simple_pipeline.evaluate(method=method, stages=None, **KWARGS)
    for stage_name in simple_pipeline.stages.keys():
        stage = simple_pipeline.stages[stage_name]
        helpers.assert_simple_logs(stage, method)
        helpers.assert_simple_output(stage, method)


@pytest.mark.e2e
def test_simple_pipeline_stages(simple_pipeline):
    stages = ["run_1", "fit_2"]
    simple_pipeline.evaluate(method="run", stages=stages, **KWARGS)
    for stage_name in stages:
        stage = simple_pipeline.stages[stage_name]
        helpers.assert_simple_logs(stage, "run")
        helpers.assert_simple_output(stage, "run")


@pytest.mark.e2e
@pytest.mark.parametrize(
    "kwargs", [{}, {"run_1": {"key1": "dummy", "key2": {"key3": "dummy"}}}]
)
def test_simple_pipeline_kwargs(simple_pipeline, kwargs):
    stage = simple_pipeline.stages["run_1"]
    stage.evaluate(method="run", stages=["run_1"], **STAGE_KWARGS, **kwargs)
    helpers.assert_simple_logs(stage, "run")
    helpers.assert_simple_output(stage, "run", **kwargs)  # TODO: check kwargs


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_pipeline_method(parallel_pipeline, method):
    parallel_pipeline.evaluate(method=method, stages=None, **KWARGS)
    for stage_name in parallel_pipeline.stages.keys():
        stage = parallel_pipeline.stages[stage_name]
        helpers.assert_parallel_logs(stage, method)
        helpers.assert_parallel_output(stage, method)


@pytest.mark.e2e
def test_parallel_pipeline_stages(parallel_pipeline):
    stages = ["run_1", "fit_2"]
    parallel_pipeline.evaluate(method="run", stages=stages, **KWARGS)
    for stage_name in stages:
        stage = parallel_pipeline.stages[stage_name]
        helpers.assert_parallel_logs(stage, "run")
        helpers.assert_parallel_output(stage, "run")


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_simple_stage_method(simple_pipeline, method):
    for stage in simple_pipeline.stages.values():
        if method not in stage.skip:
            stage.evaluate(method=method, **STAGE_KWARGS)
            helpers.assert_simple_logs(stage, method)
            helpers.assert_simple_output(stage, method)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "kwargs", [{}, {"key1": "dummy", "key2": {"key3": "dummy"}}]
)
def test_simple_stages_kwargs(simple_pipeline, kwargs):
    stage = simple_pipeline.stages["run_1"]
    stage.evaluate(method="run", **STAGE_KWARGS, **kwargs)
    helpers.assert_simple_logs(stage, "run")
    helpers.assert_simple_output(stage, "run", **kwargs)


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_stage_method(parallel_pipeline, method):
    for stage in parallel_pipeline.stages.values():
        stage.evaluate(method=method, **STAGE_KWARGS)
        helpers.assert_parallel_logs(stage, method)
        helpers.assert_parallel_output(stage, method)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "submodel",
    [
        [None, None],
        [{"sex_id": 1, "age_group_id": 1}, {"param1": 1, "param2": 1}],
    ],
)
@pytest.mark.parametrize("collect", [None, True, False])
def test_parallel_stage_submodels(parallel_pipeline, submodel, collect):
    subsets, paramsets = submodel
    stage = parallel_pipeline.stages["run_1"]
    stage.evaluate(
        method="run",
        **{
            **STAGE_KWARGS,
            "subsets": subsets,
            "paramsets": paramsets,
            "collect": collect,
        },
    )
    helpers.assert_parallel_logs(stage, "run", subsets, paramsets, collect)
    helpers.assert_parallel_output(stage, "run", subsets, paramsets, collect)
