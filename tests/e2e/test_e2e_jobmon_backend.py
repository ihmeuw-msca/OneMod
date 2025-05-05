"""End-to-end tests for Jobmon backend on dummy cluster.

Since dummy cluster doesn't actually evaluate tasks, these tests just
make sure the workflows finishes successfully.

These tests take a long time, e.g., test_simple_pipeline() took 2m 11.6s

"""

import pytest

try:
    from jobmon.client.api import Tool

    from onemod.backend.jobmon_backend import add_tasks_to_workflow
except ImportError:
    pass


KWARGS = {
    "backend": "jobmon",
    "cluster": "dummy",
    "resources": {"tool_resources": {"dummy": {"queue": "null.q"}}},
    "python": None,
    "task_prefix": "me_1234",
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


@pytest.mark.e2e
@pytest.mark.requires_jobmon
def test_simple_pipeline_add_tasks_to_workflow(simple_pipeline):
    tool = Tool(name="test_run_simple_pipeline")
    tool.set_default_cluster_name("dummy")
    tool.set_default_compute_resources_from_dict("dummy", {"queue": "null.q"})
    workflow = tool.create_workflow(name="test_run_workflow")
    add_tasks_to_workflow(
        model=simple_pipeline,
        workflow=workflow,
        method="run",
        stages=["run_1", "fit_2"],
        **KWARGS,
    )
    workflow.bind()
    workflow.run()


@pytest.mark.e2e
@pytest.mark.requires_jobmon
def test_simple_pipeline_add_tasks_to_workflow_multiple_models(
    simple_pipeline, second_simple_pipeline
):
    tool = Tool(name="test_run_simple_pipeline")
    tool.set_default_cluster_name("dummy")
    tool.set_default_compute_resources_from_dict("dummy", {"queue": "null.q"})
    workflow = tool.create_workflow(name="test_run_workflow")
    add_tasks_to_workflow(
        model=simple_pipeline,
        workflow=workflow,
        method="run",
        stages=["run_1", "fit_2"],
        **KWARGS,
    )
    # Same tasks with a "different" ME/pipeline
    add_tasks_to_workflow(
        model=second_simple_pipeline,
        workflow=workflow,
        method="run",
        stages=["run_1", "fit_2"],
        **(KWARGS | {"task_prefix": "me_1235"}),
    )
    workflow.bind()
    for task in workflow.tasks.values():
        for upstream_task in task.upstream_tasks:
            # Check task prefixes are always identical for upstreams
            assert task.name[:7] == upstream_task.name[:7]
    workflow.run()


@pytest.mark.e2e
@pytest.mark.requires_jobmon
def test_parallel_pipeline_add_tasks_to_workflow(parallel_pipeline):
    tool = Tool(name="test_run_parallel_pipeline")
    tool.set_default_cluster_name("dummy")
    tool.set_default_compute_resources_from_dict("dummy", {"queue": "null.q"})
    workflow = tool.create_workflow(name="test_run_workflow")
    add_tasks_to_workflow(
        model=parallel_pipeline,
        workflow=workflow,
        method="run",
        stages=["run_1", "fit_2"],
        **KWARGS,
    )
    workflow.bind()
    workflow.run()


@pytest.mark.e2e
@pytest.mark.requires_jobmon
def test_parallel_pipeline_add_tasks_to_workflow_multiple_models(
    parallel_pipeline, second_parallel_pipeline
):
    tool = Tool(name="test_run_parallel_pipeline")
    tool.set_default_cluster_name("dummy")
    tool.set_default_compute_resources_from_dict("dummy", {"queue": "null.q"})
    workflow = tool.create_workflow(name="test_run_workflow")
    add_tasks_to_workflow(
        model=parallel_pipeline,
        workflow=workflow,
        method="run",
        stages=["run_1", "fit_2"],
        **KWARGS,
    )
    # Same tasks with a "different" ME/pipeline
    add_tasks_to_workflow(
        model=second_parallel_pipeline,
        workflow=workflow,
        method="run",
        stages=["run_1", "fit_2"],
        **(KWARGS | {"task_prefix": "me_1235"}),
    )
    workflow.bind()
    for task in workflow.tasks.values():
        for upstream_task in task.upstream_tasks:
            # Check task prefixes are always identical for upstreams
            assert task.name[:7] == upstream_task.name[:7]
    workflow.run()
