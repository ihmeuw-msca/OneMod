"""Functions to run pipelines and stages locally."""

from typing import Any, Literal

from pydantic import validate_call

from onemod.backend.utils import (
    check_input_exists,
    check_method,
    collect_results,
)
from onemod.dtypes import UniqueList
from onemod.pipeline import Pipeline
from onemod.stage import Stage


@validate_call
def evaluate_local(
    model: Pipeline | Stage,
    method: Literal["run", "fit", "predict", "collect"],
    stages: UniqueList[str] | None = None,
    subsets: dict[str, Any | list[Any]] | None = None,
    paramsets: dict[str, Any | list[Any]] | None = None,
    collect: bool | None = None,
    **kwargs,
) -> None:
    """Evaluate pipeline or stage method locally.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance.
    method : {'run', 'fit', 'predict', 'collect}
        Name of method to evaluate.
    **kwargs
        Additional keyword arguments passed to stage methods. If `model`
        is a `Pipeline` instance, use format`stage={arg_name: arg_value}`.

    Pipeline Parameters
    -------------------
    stages : list of str, optional
        Names of stages to evaluate if `model` is a `Pipeline` instance.
        If None, evaluate all pipeline stages. Default is None.

    Stage Parameters
    ----------------
    subsets : dict, optional
        Submodel data subsets to evaluate if `model` is a `Stage`
        instance. If None, evaluate all data subsets. Default is None.
    paramsets : dict, optional
        Submodel parameter sets to evaluate if `model` is a `Stage`
        instance. If None, evaluate all parameter sets. Default is None.
    collect : bool, optional
        Whether to collect submodel results if `model` is a `Stage`
        instance. If `subsets` and `paramsets` are both None, default is
        True, otherwise default is False.

    """
    check_method(model, method)
    check_input_exists(model, stages)

    if isinstance(model, Pipeline):
        _evaluate_pipeline(model, method, stages, **kwargs)
    else:
        _evaluate_stage(model, method, subsets, paramsets, collect, **kwargs)


def _evaluate_pipeline(
    pipeline: Pipeline, method: str, stages: list[str] | None, **kwargs
) -> None:
    """Evaluate pipeline method locally."""
    for stage_name in pipeline.get_execution_order(stages):
        stage = pipeline.stages[stage_name]
        if method not in stage.skip:
            _evaluate_stage(stage, method, **kwargs.get(stage_name, {}))


def _evaluate_stage(
    stage: Stage,
    method: str,
    subsets: dict[str, Any | list[Any]] | None = None,
    paramsets: dict[str, Any | list[Any]] | None = None,
    collect: bool | None = None,
    **kwargs,
) -> None:
    """Evaluate stage method locally."""
    if method == "collect":
        stage.collect()
    else:
        stage_method = stage.__getattribute__(f"_{method}")
        if stage.has_submodels:
            for subset, paramset in stage.get_submodels(subsets, paramsets):
                if subset is None:
                    stage_method(paramset=paramset, **kwargs)
                elif paramset is None:
                    stage_method(subset=subset, **kwargs)
                else:
                    stage_method(subset=subset, paramset=paramset, **kwargs)

            if collect_results(stage, method, subsets, paramsets, collect):
                stage.collect()
        else:
            stage_method(**kwargs)
