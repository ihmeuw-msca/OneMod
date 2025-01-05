"""Functions for working with crossby and params."""

from itertools import product
from typing import Any

from pandas import DataFrame

from onemod.config import StageConfig


def create_params(crossby: tuple[str, ...], config: StageConfig) -> DataFrame:
    """Create parameter sets from crossby."""
    param_dict = {}
    for param_name in crossby:
        param_values = config[param_name]
        if isinstance(param_values, (list, set, tuple)):
            param_dict[param_name] = param_values
        else:
            raise ValueError(
                f"Crossby param '{param_name}' must be a list, set, or tuple"
            )

    params = DataFrame(
        list(product(*param_dict.values())), columns=list(param_dict.keys())
    )
    params.sort_values(by=crossby)
    params["param_id"] = params.index

    return params[["param_id", *crossby]]


def get_params(params: DataFrame, param_id: int) -> dict[str, Any]:
    """Get parameter values corresponding to parameter set ID."""
    params = params.query("param_id == @param_id").drop(columns=["param_id"])
    return {
        str(param_name): param_value.item()
        for param_name, param_value in params.items()
    }
