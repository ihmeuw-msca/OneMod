"""Functions for working with crossby and params."""

from itertools import product
from typing import Any

import polars as pl

from onemod.config import ModelConfig


def create_params(config: ModelConfig) -> pl.DataFrame | None:
    """Create parameter sets from crossby."""
    param_dict = {
        param_name: param_values
        for param_name in config.crossable_params
        if isinstance(param_values := config[param_name], set)
    }
    if len(param_dict) == 0:
        return None

    crossby = list(param_dict.keys())
    params = pl.DataFrame(
        [list(param_set) for param_set in product(*param_dict.values())],
        schema=crossby,
        orient="row",
    )

    params = params.with_row_index(name="param_id")
    return params.select(["param_id", *crossby])


def get_params(params: pl.DataFrame, param_id: int) -> dict[str, Any]:
    params = params.filter(pl.col("param_id") == param_id).drop("param_id")
    return {str(col): params[col][0] for col in params.columns}
