"""Functions to run pipelines and stages locally."""

from typing import Literal

from pydantic import validate_call

from onemod.backend.utils import check_input, check_method
from onemod.pipeline import Pipeline
from onemod.stage import ModelStage, Stage


@validate_call
def evaluate(
    model: Pipeline | Stage,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    stages: set[str] | None = None,
    subset_id: int | None = None,
    param_id: int | None = None,
) -> None:
    """Evaluate pipeline method locally.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance to evaluate.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    stages : set of str or None, optional
        Names of stages to evaluate if `model` is a `Pipeline` instance.
        If None, evaluate entire pipeline. Default is None.
    subset_id : int, optional
        Submodel data subset ID if `model` is a `ModelStage` instance.
        If None, evaluate all data subsets. Default is None.
    param_id : int, optional
        Submodel parameter set ID if `model` is a `ModelStage` instance.
        If None, evaluate all parameter sets. Default is None.

    """
    check_method(model, method, backend="local")
    check_input(model, stages)
    if isinstance(model, Pipeline):
        for stage_name in model.get_execution_order(stages):
            stage = model.stages[stage_name]
            if method not in stage.skip:
                _evaluate_stage(stage, method)
    else:
        _evaluate_stage(model, method, subset_id, param_id)


def _evaluate_stage(
    stage: Stage,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    subset_id: int | None = None,
    param_id: int | None = None,
) -> None:
    """Evaluate pipeline method locally.

    Parameters
    ----------
    stage : Stage
        Stage instance to evaluate.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    subset_id : int, optional
        Submodel data subset ID if `model` is a `ModelStage` instance.
        If None, evaluate all data subsets. Default is None
    param_id : int, optional
        Submodel parameter set ID if `model` is a `ModelStage` instance.
        If None, evaluate all parameter sets. Default is None.

    """
    if isinstance(stage, ModelStage):
        if method == "collect":
            stage.collect()
        else:
            if subset_id is None and param_id is None:
                for subset_id in stage.subset_ids or [None]:
                    for param_id in stage.param_ids or [None]:
                        stage.__getattribute__(method)(subset_id, param_id)
                if method in stage.collect_after:
                    stage.collect()
            else:
                stage.__getattribute__(method)(subset_id, param_id)
    else:
        stage.__getattribute__(method)()
