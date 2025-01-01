"""Run pipeline through local and Jobmon backends.

Since dummy cluster doesn't actually evaluate tasks, the Jobmon tests
just make sure the workflow finishes successfully.

"""

import pytest
import tests.helpers.orchestration_helpers as helpers

SETUP_PIPELINE = {
    "simple": helpers.setup_simple_pipeline,
    "parallel": helpers.setup_parallel_pipeline,
}

ASSERT_LOGS = {
    "simple": helpers.assert_simple_logs,  # type: ignore
    "parallel": helpers.assert_parallel_logs,  # type: ignore
}

ASSERT_OUTPUT = {
    "simple": helpers.assert_simple_output,  # type: ignore
    "parallel": helpers.assert_parallel_output,  # type: ignore
}


@pytest.mark.e2e
@pytest.mark.parametrize("pipeline_type", ["simple", "parallel"])
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stage_names", [None, ["run_1", "fit_2"]])
def test_local_pipeline(tmp_path, pipeline_type, method, stage_names):
    # Setup the pipeline
    pipeline = SETUP_PIPELINE[pipeline_type](tmp_path)

    # Run pipeline through Jobmon backend on dummy cluster
    pipeline.evaluate(method=method, stages=stage_names, backend="local")

    # Check logs and output
    stage_names = stage_names or pipeline.stages.keys()
    for stage_name in stage_names or pipeline.stages:
        stage = pipeline.stages[stage_name]
        ASSERT_LOGS[pipeline_type](stage, method)
        ASSERT_OUTPUT[pipeline_type](stage, method)


@pytest.mark.e2e
@pytest.mark.parametrize("pipeline_type", ["simple", "parallel"])
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
@pytest.mark.parametrize("stages", [None, ["run_1", "fit_2"]])
def test_jobmon_pipeline(tmp_path, pipeline_type, method, stages):
    # Setup the pipeline
    pipeline = SETUP_PIPELINE[pipeline_type](tmp_path)

    # Run pipeline through Jobmon backend on dummy cluster
    pipeline.evaluate(
        method=method,
        stages=stages,
        backend="jobmon",
        cluster="dummy",
        resources={"tool_resources": {"dummy": {"queue": "null.q"}}},
    )
