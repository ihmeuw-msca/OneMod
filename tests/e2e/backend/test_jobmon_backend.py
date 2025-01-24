"""End-to-end tests for Jobmon backend on dummy cluster.

Since dummy cluster doesn't actually evaluate tasks, these tests just
make sure the workflows finishes successfully.

"""
# TODO: Write tests for subsets/paramsets

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
        if stage.name == "run_1":
            stage.evaluate(method=method, **JOBMON_ARGS)
        else:
            if method not in stage.skip:
                # Input from upstream stages won't exist since dummy cluster
                # doesn't evaluate tasks, so check_input should raise error
                with pytest.raises(FileNotFoundError):
                    stage.evaluate(method=method, **JOBMON_ARGS)


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_stage(parallel_pipeline, method):
    # Inputs are paths or upstream stage output directories, so check_input
    # shouldn't raise an error
    for stage in parallel_pipeline.stages.values():
        stage.evaluate(method=method, **JOBMON_ARGS)
