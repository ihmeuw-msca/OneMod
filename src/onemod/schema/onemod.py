from typing import Any

from modrover.globals import model_type_dict
from pydantic import field_validator

from onemod.schema.base import Config
from onemod.schema.stages import (
    EnsembleConfig,
    RegmodSmoothConfig,
    RoverCovselConfig,
    WeaveConfig,
)


class OneModConfig(Config):
    input_path: str
    col_id: list[str]
    col_obs: str
    col_pred: str
    col_holdout: list[str]
    col_test: str
    id_subsets: dict[str, list[Any]] = {}

    mtype: str
    weights: str

    rover_covsel: RoverCovselConfig | None = None
    regmod_smooth: RegmodSmoothConfig | None = None
    weave: dict[str, WeaveConfig] | None = None
    ensemble: EnsembleConfig | None = None

    @field_validator("mtype")
    @classmethod
    def valid_model_type(cls, model_type: str) -> str:
        assert (
            model_type in model_type_dict
        ), f"model_type must be one of {model_type_dict.keys()}"
        return model_type
