"""End-to-end tests for Jobmon backend on dummy cluster.

Since dummy cluster doesn't actually evaluate tasks, these tests just
make sure the workflows finishes successfully.

These tests take a long time, e.g., test_simple_pipeline() took 2m 11.6s

"""

import pytest

KWARGS = {
    "backend": "jobmon",
    "cluster": "dummy",
    "resources": {"tool_resources": {"dummy": {"queue": "null.q"}}},
    "python": None,
    "template_prefix": "jobmon_e2e_testing",
    "max_attempts": 3,
}

STAGE_KWARGS = {**KWARGS, "subsets": None, "paramsets": None, "collect": None}


@pytest.mark.e2e
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_simple_pipeline_method(simple_pipeline, method):
    simple_pipeline.evaluate(method=method, stages=None, **KWARGS)


@pytest.mark.e2e
@pytest.mark.requires_jobmon
def test_simple_pipeline_stages(simple_pipeline):
    simple_pipeline.evaluate(method="run", stages=["run_1", "fit_2"], **KWARGS)


@pytest.mark.e2e
@pytest.mark.requires_jobmon
@pytest.mark.parametrize(
    "kwargs", [{}, {"run_1": {"key1": "dummy", "key2": {"key3": "dummy"}}}]
)
def test_simple_pipeline_kwargs(simple_pipeline, kwargs):
    stage = simple_pipeline.stages["run_1"]
    stage.evaluate(method="run", stages=["run_1"], **STAGE_KWARGS, **kwargs)


@pytest.mark.e2e
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_pipeline_method(parallel_pipeline, method):
    parallel_pipeline.evaluate(method=method, stages=None, **KWARGS)


@pytest.mark.e2e
@pytest.mark.requires_jobmon
def test_parallel_pipeline_stages(parallel_pipeline):
    parallel_pipeline.evaluate(
        method="run", stages=["run_1", "fit_2"], **KWARGS
    )


@pytest.mark.e2e
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_simple_stage_method(simple_pipeline, method):
    for stage in simple_pipeline.stages.values():
        if stage.name == "run_1":
            stage.evaluate(method=method, **STAGE_KWARGS)
        else:
            if method not in stage.skip:
                # Input from upstream stages won't exist since dummy cluster
                # doesn't evaluate tasks, so check_input should raise error
                with pytest.raises(FileNotFoundError):
                    stage.evaluate(method=method, **STAGE_KWARGS)


@pytest.mark.e2e
@pytest.mark.requires_jobmon
@pytest.mark.parametrize(
    "kwargs", [{}, {"key1": "dummy", "key2": {"key3": "dummy"}}]
)
def test_simple_stages_kwargs(simple_pipeline, kwargs):
    stage = simple_pipeline.stages["run_1"]
    stage.evaluate(method="run", **STAGE_KWARGS, **kwargs)


@pytest.mark.e2e
@pytest.mark.requires_jobmon
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_parallel_stage_method(parallel_pipeline, method):
    # Inputs are paths or upstream stage output directories,
    # so check_input shouldn't raise an error
    for stage in parallel_pipeline.stages.values():
        stage.evaluate(method=method, **STAGE_KWARGS)


@pytest.mark.e2e
@pytest.mark.requires_jobmon
@pytest.mark.parametrize(
    "submodel",
    [
        [None, None],
        [{"sex_id": 1, "age_group_id": 1}, {"param1": 1, "param2": 1}],
    ],
)
@pytest.mark.parametrize("collect", [True, False])
def test_parallel_stage_submodels(parallel_pipeline, submodel, collect):
    stage = parallel_pipeline.stages["run_1"]
    stage.evaluate(
        method="run",
        **{
            **STAGE_KWARGS,
            "subsets": submodel[0],
            "paramsets": submodel[1],
            "collect": collect,
        },
    )
