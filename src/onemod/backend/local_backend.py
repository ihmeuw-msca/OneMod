"""Functions to run pipelines and stages locally."""

from itertools import product
from typing import Literal

from pydantic import validate_call

from onemod.pipeline import Pipeline
from onemod.stage import ModelStage, Stage


@validate_call
def evaluate_local(
    model: Pipeline | Stage,
    method: Literal["run", "fit", "predict"] = "run",
    stages: set[str] | None = None,
    **kwargs,
) -> None:
    """Evaluate pipeline or stage method locally.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    stages : set of str or None, optional
        Names of stages to evaluate if `model` is a pipeline instance.
        If None, evaluate entire pipeline. Default is None.

    Other Parameters
    ----------------
    subset_id : int, optional
        Submodel data subset ID. Only used for model stages.
    param_id : int, optional
        Submodel parameter set ID. Only used for model stages.

    """
    if isinstance(model, Pipeline):
        for stage_name in model.get_execution_order(stages):
            stage = model.stages[stage_name]
            if method not in stage.skip:
                _evaluate_stage(stage, method)
    else:
        _evaluate_stage(model, method, **kwargs)


def _evaluate_stage(
    stage: Stage, method: Literal["run", "fit", "predict"]
) -> None:
    """Evaluate stage method locally."""
    if isinstance(stage, ModelStage):
        subset_ids = stage.subset_ids or [None]
        param_ids = stage.param_ids or [None]
        for subset_id, param_id in product(subset_ids, param_ids):
            stage.__getattribute__(method)(subset_id, param_id)
        if method in stage.collect_after:
            stage.collect()
    else:
        stage.evaluate(method=method)
