"""Utility functions used by all backends."""
# TODO: Simplify pipeline class by moving validation method here

from onemod.pipeline import Pipeline
from onemod.stage import Stage


def check_method(model: Pipeline | Stage, method: str) -> None:
    if isinstance(model, Pipeline):
        if method == "collect":
            raise ValueError(
                "Method 'collect' cannot be called on a pipeline instance"
            )
    else:
        if method == "collect":
            if model.has_submodels:
                if len(model.collect_after) == 0:
                    raise ValueError(
                        "Method 'collect' cannot be called on stage with empty collect_after"
                    )
            else:
                raise ValueError(
                    "Method 'collect' cannot be called on a stage without submodels"
                )
        elif method in model.skip:
            raise ValueError(
                f"Stage '{model.name}' skips the '{method}' method"
            )


def check_input_exists(
    model: Pipeline | Stage, stages: list[str] | None = None
) -> None:
    if isinstance(model, Pipeline):
        # Check input already exists for any upstream stage not being evaluated
        stages_being_evaluated = stages or model.stages.keys()
        for stage_name in stages_being_evaluated:
            if stage_name not in model.stages:
                raise ValueError(f"Stage '{stage_name}' not found in pipeline")
            stage = model.stages[stage_name]
            stage.input.check_exists(
                upstream_stages=[
                    dependency
                    for dependency in stage.dependencies
                    if dependency not in stages_being_evaluated
                ]
            )
    else:
        model.input.check_exists()
