"""Run pipeline and stages through local backend."""
# TODO: Write tests for id_subsets

import pytest
import tests.helpers.orchestration_helpers as helpers


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["run_1", "fit_2"]])
def test_simple_pipeline(simple_pipeline, method, stages):
    simple_pipeline.evaluate(method=method, stages=stages)
    for stage_name in stages or simple_pipeline.stages.keys():
        stage = simple_pipeline.stages[stage_name]
        helpers.assert_simple_logs(stage, method)
        helpers.assert_simple_output(stage, method)


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["run_1", "fit_2"]])
def test_parallel_pipeline(parallel_pipeline, method, stages):
    parallel_pipeline.evaluate(method=method, stages=stages)
    for stage_name in stages or parallel_pipeline.stages.keys():
        stage = parallel_pipeline.stages[stage_name]
        helpers.assert_parallel_logs(stage, method)
        helpers.assert_parallel_output(stage, method)


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_simple_stage(simple_pipeline, method):
    for stage in simple_pipeline.stages.values():
        if method not in stage.skip:
            stage.evaluate(method=method)
            helpers.assert_simple_logs(stage, method)
            helpers.assert_simple_output(stage, method)


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_stage(parallel_pipeline, method):
    for stage in parallel_pipeline.stages.values():
        stage.evaluate(method=method)
        helpers.assert_parallel_logs(stage, method)
        helpers.assert_parallel_output(stage, method)
