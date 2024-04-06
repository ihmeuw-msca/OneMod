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
    max_attempts: int = 3
    max_batch: int = -1
    id_subsets: dict[str, list[Any]] = {}
    mtype: str

    rover_covsel: RoverCovselConfig | None = None
    regmod_smooth: RegmodSmoothConfig | None = None
    weave: WeaveConfig | None = None
    ensemble: EnsembleConfig | None = None

    def __init__(self, **data):
        super().__init__(**data)
        # Pass global attributes to children
        global_vals = {
            "input_path": self.input_path,
            "col_id": self.col_id,
            "col_obs": self.col_obs,
            "col_pred": self.col_pred,
            "col_holdout": self.col_holdout,
            "col_test": self.col_test,
            "max_attempts": self.max_attempts,
            "max_batch": self.max_batch,
            "mtype": self.mtype,
        }

        child_models = [
            self.rover_covsel,
            self.regmod_smooth,
            self.weave,
            self.ensemble,
        ]

        for child_model in child_models:
            if child_model:
                # Store parent args on the child models, can be accessed if necessary
                child_model.parent_args = global_vals
                child_model.inherit()

    @property
    def extra_fields(self) -> set[str]:
        return set(self.__dict__) - set(self.model_fields)

    @field_validator("mtype")
    @classmethod
    def valid_model_type(cls, model_type: str) -> str:
        assert (
            model_type in model_type_dict
        ), f"model_type must be one of {model_type_dict.keys()}"
        return model_type
