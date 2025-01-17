"""Functions to run pipelines and stages locally."""

from typing import Literal

from pydantic import validate_call

from onemod.backend.utils import check_input, check_method
from onemod.dtypes import UniqueList
from onemod.pipeline import Pipeline
from onemod.stage import Stage


@validate_call
def evaluate(
    model: Pipeline | Stage,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    stages: UniqueList[str] | None = None,
    subset_id: int | None = None,
    param_id: int | None = None,
    **kwargs,
) -> None:
    """Evaluate pipeline method locally.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance to evaluate.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    stages : list of str or None, optional
        Names of stages to evaluate if `model` is a `Pipeline` instance.
        If None, evaluate entire pipeline. Default is None.
    subset_id : int, optional
        Submodel data subset ID if `model` is a `Stage` instance.
        If None, evaluate all data subsets. Default is None.
    param_id : int, optional
        Submodel parameter set ID if `model` is a `Stage` instance.
        If None, evaluate all parameter sets. Default is None.

    """
    check_method(model, method, backend="local")
    check_input(model, stages)
    if isinstance(model, Pipeline):
        for stage_name in model.get_execution_order(stages):
            stage = model.stages[stage_name]
            if method not in stage.skip:
                _evaluate_stage(stage, method, **kwargs)
    else:
        _evaluate_stage(model, method, subset_id, param_id, **kwargs)


def _evaluate_stage(
    stage: Stage,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    subset_id: int | None = None,
    param_id: int | None = None,
    **kwargs,
) -> None:
    """Evaluate pipeline method locally.

    Parameters
    ----------
    stage : Stage
        Stage instance to evaluate.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    subset_id : int, optional
        Submodel data subset ID. If None, evaluate all data subsets.
        Default is None
    param_id : int, optional
        Submodel parameter set ID. If None, evaluate all parameter sets.
        Default is None.

    Notes
    -----
    If stage uses both `groupby` and `crossby`, both `subset_id` and
    `param_id` must be passed to evaluate in individual submodels,
    otherwise all submodels will be evaluated.

    """
    if method == "collect":
        stage.collect()
    else:
        stage_method = stage.__getattribute__(method)
        if stage.has_submodels:
            eval_submodel = False
            if subset_id is not None:
                if len(stage.groupby) == 0:
                    raise ValueError(
                        f"Stage '{stage.name}' does not use groupby attribute"
                    )
                eval_submodel = True
            if param_id is not None:
                if len(stage.crossby) == 0:
                    raise ValueError(
                        f"Stage '{stage.name}' does not use crossby attribute"
                    )
                eval_submodel = True

            if eval_submodel:
                stage_method(subset_id, param_id, **kwargs)
            else:
                for subset_id, param_id in stage.submodel_ids:
                    stage_method(subset_id, param_id, **kwargs)
                if method in stage.collect_after:
                    stage.collect()
        else:
            stage_method(**kwargs)
