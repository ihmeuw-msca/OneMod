"""Unit tests for backend helper functions."""

import pytest
from tests.helpers.orchestration_helpers import ParallelStage

import onemod.backend.utils as utils


def test_check_method_pipeline(parallel_pipeline):
    with pytest.raises(
        ValueError,
        match="Method 'collect' cannot be called on a pipeline instance",
    ):
        utils.check_method(parallel_pipeline, "collect")


def test_check_method_stage_collect_simple(simple_pipeline):
    with pytest.raises(
        ValueError,
        match="Method 'collect' cannot be called on a stage without submodels",
    ):
        utils.check_method(simple_pipeline.stages["run_1"], "collect")


def test_check_method_stage_collect_parallel_empty_collect_after():
    class DummyStage(ParallelStage):
        _collect_after = []

    dummy_stage = DummyStage(name="dummy", groupby=["sex_id"])

    with pytest.raises(
        ValueError,
        match="Method 'collect' cannot be called on stage with empty collect_after",
    ):
        utils.check_method(dummy_stage, "collect")


@pytest.mark.parametrize("args", [["fit_2", "predict"], ["predict_3", "fit"]])
def test_check_method_stage_skip(simple_pipeline, args):
    with pytest.raises(
        ValueError, match=f"Stage '{args[0]}' skips the '{args[1]}' method"
    ):
        utils.check_method(simple_pipeline.stages[args[0]], args[1])


def test_check_input_stage(simple_pipeline):
    with pytest.raises(FileNotFoundError):
        utils.check_input_exists(simple_pipeline.stages["fit_2"])


def test_check_input_with_invalid_stage(simple_pipeline):
    with pytest.raises(ValueError):
        utils.check_input_exists(simple_pipeline, ["dummy_stage"])


def test_check_input_pipeline_stages(simple_pipeline):
    with pytest.raises(FileNotFoundError):
        utils.check_input_exists(simple_pipeline, ["predict_3"])


def test_collect_results_no_submodels(simple_pipeline):
    collect_results = utils.collect_results(
        simple_pipeline.stages["run_1"], "run", None, None, None
    )
    assert not collect_results


def test_collect_results_collect_method(parallel_pipeline):
    collect_results = utils.collect_results(
        parallel_pipeline.stages["run_1"], "collect", None, None, None
    )
    assert not collect_results


@pytest.mark.parametrize("args", [["fit_2", "predict"], ["predict_3", "fit"]])
def test_collect_results_not_in_collect_after(parallel_pipeline, args):
    collect_results = utils.collect_results(
        parallel_pipeline.stages[args[0]], args[1], None, None, None
    )
    assert not collect_results


@pytest.mark.parametrize("collect", [True, False])
def test_collect_results_collect_arg(parallel_pipeline, collect):
    collect_results = utils.collect_results(
        parallel_pipeline.stages["run_1"], "run", None, None, collect
    )
    assert collect_results == collect


@pytest.mark.parametrize("subsets", [None, {}])
@pytest.mark.parametrize("paramsets", [None, {}])
def test_collect_results_submodel_args(parallel_pipeline, subsets, paramsets):
    collect_results = utils.collect_results(
        parallel_pipeline.stages["run_1"], "run", subsets, paramsets, None
    )
    assert collect_results == (subsets is None and paramsets is None)
