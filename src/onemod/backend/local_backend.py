"""Functions to run pipelines and stages locally."""

from typing import Any, Literal

from onemod.backend.utils import check_input_exists, check_method
from onemod.dtypes import UniqueList
from onemod.pipeline import Pipeline
from onemod.stage import Stage


def evaluate(
    model: Pipeline | Stage,
    method: Literal["run", "fit", "predict", "collect"],
    method_args: dict[str, Any | dict[str, Any]] | None = None,
    stages: UniqueList[str] | None = None,
    subsets: dict[str, Any | list[Any]] | None = None,
    paramsets: dict[str, Any | list[Any]] | None = None,
    collect: bool = False,
) -> None:
    """Evaluate pipeline or stage method locally.

    Parameters
    ----------
    model : Pipeline or Stage
        Pipeline or stage instance.
    method : {'run', 'fit', 'predict', 'collect}
        Name of method to evaluate.
    method_args : dict, optional
        Additional keyword arguments passed to stage methods. If `model`
        is a `Pipeline` instance, use format `{stage_name: {arg_name: arg_value}}`.
    stages : list of str, optional
        Names of stages to evaluate if `model` is a `Pipeline` instance.
        If None, evaluate all pipeline stages. Default is None.
    subsets : dict, optional
        Submodel data subsets to evaluate if `model` is a `Stage`
        instance. If None, evaluate all data subsets. Default is None.
    paramsets : dict, optional
        Submodel parameter sets to evaluate if `model` is a `Stage`
        instance. If None, evaluate all parameter sets. Default is None.
    collect : bool, optional
        Collect submodel results if `model` is a `Stage` instance and
        `subsets` and `paramsets` are not both None. Default is False.
        If `subsets` and `paramsets` are both None, this parameter is
        ignored and submodel results are always collected.

    """
    check_method(model, method)
    check_input_exists(model, stages)
    if method_args is None:
        method_args = {}

    if isinstance(model, Pipeline):
        _evaluate_pipeline(model, method, method_args, stages)
    else:
        _evaluate_stage(model, method, method_args, subsets, paramsets, collect)


def _evaluate_pipeline(
    pipeline: Pipeline,
    method: str,
    method_args: dict[str, dict[str, Any]],
    stages: list[str] | None,
) -> None:
    """Evaluate pipeline method locally."""
    for stage_name in pipeline.get_execution_order(stages):
        stage = pipeline.stages[stage_name]
        if method not in stage.skip:
            _evaluate_stage(stage, method, method_args.get(stage_name, {}))


def _evaluate_stage(
    stage: Stage,
    method: str,
    method_args: dict[str, Any],
    subsets: dict[str, Any | list[Any]] | None = None,
    paramsets: dict[str, Any | list[Any]] | None = None,
    collect: bool = False,
) -> None:
    """Evaluate stage method locally."""
    if method == "collect":
        stage.collect()
    else:
        stage_method = stage.__getattribute__(f"_{method}")
        if stage.has_submodels:
            for subset, paramset in stage.get_submodels(subsets, paramsets):
                stage_method(subset=subset, paramset=paramset, **method_args)

            if method in stage.collect_after:
                if collect or (subsets is None and paramsets is None):
                    stage.collect()
        else:
            stage_method(**method_args)
