"""Functions for working with crossby and params."""

from itertools import product
from typing import Any

from pandas import DataFrame

from onemod.config import ModelConfig


def create_params(config: ModelConfig) -> DataFrame | None:
    """Create parameter sets from crossby."""
    param_dict = {
        param_name: param_values
        for param_name in config.crossable_params
        if isinstance(param_values := config[param_name], set)
    }
    if len(param_dict) == 0:
        return None

    params = DataFrame(
        [param_set for param_set in product(*param_dict.values())],
        columns=(crossby := list(param_dict.keys())),
    )
    params["param_id"] = params.index

    return params[["param_id", *crossby]]


def get_params(params: DataFrame, param_id: int) -> dict[str, Any]:
    params = params.query("param_id == @param_id").drop(columns=["param_id"])
    return {
        str(param_name): param_value.item()
        for param_name, param_value in params.items()
    }
