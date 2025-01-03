"""Run pipeline and stages through Jobmon backend.

Since dummy cluster doesn't actually evaluate tasks, the Jobmon tests
just make sure the workflow finishes successfully.

"""
# TODO: Write tests for id_subsets

import pytest

JOBMON_ARGS = {
    "backend": "jobmon",
    "cluster": "dummy",
    "resources": {"tool_resources": {"dummy": {"queue": "null.q"}}},
}


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["run_1", "fit_2"]])
def test_simple_pipeline(simple_pipeline, method, stages):
    simple_pipeline.evaluate(method=method, stages=stages, **JOBMON_ARGS)


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["run_1", "fit_2"]])
def test_parallel_pipeline(parallel_pipeline, method, stages):
    parallel_pipeline.evaluate(method=method, stages=stages, **JOBMON_ARGS)


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_simple_stage(simple_pipeline, method):
    for stage in simple_pipeline.stages.values():
        if method not in stage.skip:
            stage.evaluate(method=method, **JOBMON_ARGS)


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_stage(parallel_pipeline, method):
    for stage in parallel_pipeline.stages.values():
        stage.evaluate(method=method, **JOBMON_ARGS)
