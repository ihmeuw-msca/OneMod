"""Functions for working with crossby and params."""

from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from onemod.config import CrossedConfig


def create_params(config: CrossedConfig) -> pd.DataFrame | None:
    """Create parameter sets from crossby."""
    param_dict = {
        param_name: param_values
        for param_name in config.crossable_params
        if isinstance(param_values := config[param_name], set)
    }
    if len(param_dict) == 0:
        return None
    params = pd.DataFrame(
        [param_set for param_set in product(*param_dict.values())],
        columns=(crossby := param_dict.keys()),
    )
    params["param_id"] = params.index
    return params[["param_id", *crossby]]


def get_params(params: Path | str, param_id: int) -> dict[str, Any]:
    params = (
        pd.read_csv(params)
        .query("param_id == @param_id")
        .drop(columns=["param_id"])
    )
    return {
        param_name: param_value.item()
        for param_name, param_value in params.items()
    }