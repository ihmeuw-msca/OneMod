"""Functions to run pipelines and stages locally."""

from typing import Any, Literal

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
    subsets: dict[str, Any | list[Any]] | None = None,
    paramsets: dict[str, Any | list[Any]] | None = None,
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
    subsets : dict, optional
        Submodel data subsets if `model` is a `Stage` instance.
        If None, evaluate all data subsets. Default is None.
    paramsets : dict, optional
        Submodel parameter sets if `model` is a `Stage` instance.
        If None, evaluate all parameter sets. Default is None.

    Other Parameters
    ----------------
    **kwargs
        Additional keyword arguments passed to stage methods.

    """
    check_method(model, method, backend="local")
    check_input(model, stages)
    if isinstance(model, Pipeline):
        for stage_name in model.get_execution_order(stages):
            stage = model.stages[stage_name]
            if method not in stage.skip:
                _evaluate_stage(stage, method, **kwargs)
    else:
        _evaluate_stage(model, method, subsets, paramsets, **kwargs)


def _evaluate_stage(
    stage: Stage,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    subsets: dict[str, Any | list[Any]] | None = None,
    paramsets: dict[str, Any | list[Any]] | None = None,
    collect: bool = False,
    **kwargs,
) -> None:
    """Evaluate pipeline method locally.

    If both `subsets` and `paramsets` are None, evaluate all submodels
    and collect submodel results.

    Parameters
    ----------
    stage : Stage
        Stage instance to evaluate.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    subsets : dict, optional
        Submodel data subsets. If None, evaluate all data subsets.
        Default is None.
    paramsets : dict, optional
        Submodel parameter sets. If None, evaluate all parameter sets.
        Default is None.
    collect : bool, optional
        Collect submodel results if `subsets` and `paramsets` are not
        both None. Default is False.

    Other Parameters
    ----------------
    **kwargs
        Additional keyword arguments passed to stage methods.

    """
    if method == "collect":
        stage.collect()
    else:
        stage_method = stage.__getattribute__(method)
        if stage.has_submodels:
            for subset, paramset in stage.get_submodels(subsets, paramsets):
                stage_method(subset=subset, paramset=paramset, **kwargs)

            if method in stage.collect_after:
                if collect or (subsets is None and paramsets is None):
                    stage.collect()
        else:
            stage_method(**kwargs)
