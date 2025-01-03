"""Utility functions used by all backends."""
# TODO: Simplify pipeline class by moving validation method here

import warnings

from onemod.pipeline import Pipeline
from onemod.stage import ModelStage, Stage


def check_method(model: Pipeline | Stage, method: str, backend: str) -> None:
    if isinstance(model, Stage):
        if method in model.skip:
            warnings.warn(f"{model.name} skips the '{method}' method")
            return

    if method == "collect":
        if not isinstance(model, ModelStage):
            raise ValueError(
                "Method 'collect' can only be called on instances of 'ModelStage'"
            )
        if backend == "jobmon":
            raise ValueError(
                "Method 'collect' cannot be used with 'jobmon' backend"
            )


def check_input(
    model: Pipeline | Stage, stages: set[str] | None = None
) -> None:
    if isinstance(model, Stage):
        model.input.check_exists()
    else:
        # Check input already exists if upstream stage not being evaluated
        stage_names = stages or model.stages.keys()
        for stage_name in stage_names:
            if stage_name not in model.stages:
                raise ValueError(f"Stage '{stage_name}' not found in pipeline")
            stage = model.stages[stage_name]
            stage.input.check_exists(
                upstream_stages=[
                    dep for dep in stage.dependencies if dep not in stage_names
                ]
            )
