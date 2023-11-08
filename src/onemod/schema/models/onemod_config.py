from modrover.globals import model_type_dict
from pydantic import field_validator, Field
from typing import Any, Optional

from onemod.schema.models.base import ParametrizedBaseModel
from onemod.schema.models.rover_config import RoverConfiguration
from onemod.schema.models.regmod_smooth_config import RegmodSmoothConfiguration
from onemod.schema.models.weave_config import WeaveConfiguration
from onemod.schema.models.swimr_config import SwimrConfiguration
from onemod.schema.models.ensemble_config import EnsembleConfiguration


class OneModConfig(ParametrizedBaseModel):

    input_path: str
    col_id: list[str]
    col_obs: str
    col_pred: str
    col_holdout: list[str]
    col_test: str
    col_sigma: str = ''
    max_attempts: int = 3
    max_batch: int = -1
    id_subsets: dict[str, list[Any]] = {}
    mtype: str = Field(alias="model_type")

    rover_covsel: Optional[RoverConfiguration] = None
    regmod_smooth: Optional[RegmodSmoothConfiguration] = None
    weave: Optional[WeaveConfiguration] = None
    swimr: Optional[SwimrConfiguration] = None
    ensemble: Optional[EnsembleConfiguration] = None

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
            "col_sigma": self.col_sigma,
            "max_attempts": self.max_attempts,
            "max_batch": self.max_batch,
            "mtype": self.mtype
        }

        child_models = [
            self.rover_covsel,
            self.regmod_smooth,
            self.weave,
            self.swimr,
            self.ensemble
        ]

        for child_model in child_models:
            if child_model:
                # Store parent args on the child models, can be accessed if necessary
                child_model.parent_args = global_vals

    @property
    def extra_fields(self) -> set[str]:
        return set(self.__dict__) - set(self.model_fields)

    @field_validator("mtype")
    @classmethod
    def valid_model_type(cls, model_type: str) -> str:
        assert model_type in model_type_dict, \
            f"model_type must be one of {model_type_dict.keys()}"
        return model_type
