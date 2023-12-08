from typing import Optional

from pydantic import BaseModel, ConfigDict, FilePath, ValidationError, validator
from pydantic.functional_validators import field_validator

from modrover.globals import model_type_dict


class ParametrizedBaseModel(BaseModel):
    """An extension of BaseModel that supports __getitem__ and is configured."""
    model_config = ConfigDict(extra='allow', frozen=False, validate_assignment=True)

    def __getitem__(self, item):
        return getattr(self, item)


class RoverConfiguration(ParametrizedBaseModel):

    groupby: list[str] = []
    model_type: str  # TODO: This clashes with pydantic naming conventions and will raise warnings
    cov_fixed: list[str] = []
    cov_exploring: list[str] = []
    weights: str
    holdouts: list[str] = []
    fit_args: dict = {}

    parent_args: dict = {}

    @field_validator("model_type")
    @classmethod
    def valid_model_type(cls, model_type: str) -> str:
        assert model_type in model_type_dict, \
            f"model_type must be one of {model_type_dict.keys()}"
        return model_type

    @field_validator("fit_args")
    @classmethod
    def valid_fit_args(cls, fit_args: dict):
        # TODO: Necessary or not to import and validate?
        # Could import Rover.fit and inspect the args
        return fit_args


class RegmodSmoothConfiguration(ParametrizedBaseModel):

    model_type: str
    dims: list[dict] = []
    var_groups: list[dict] = []
    weights: str
    fit_args: dict = {}
    inv_link: str
    coef_bounds: dict[str, list[float]] = {}
    lam: float = 0.0

    parent_args: dict = {}

    @field_validator("model_type")
    @classmethod
    def valid_model_type(cls, model_type: str) -> str:
        if model_type not in model_type_dict:
            raise ValidationError(
                f"model_type must be one of {model_type_dict.keys()}"
            )
        return model_type


class WeaveConfiguration(ParametrizedBaseModel):
    # TODO
    pass


class SwimrConfiguration(ParametrizedBaseModel):
    # TODO
    pass


class EnsembleConfiguration(ParametrizedBaseModel):
    # TODO
    pass


class ParentConfiguration(ParametrizedBaseModel):

    input_path: FilePath  # FilePath auto-validates that the path exists and is a file
    col_id: list[str]
    col_obs: str
    col_pred: str
    col_holdout: list[str]
    col_test: str
    col_sigma: str = ''
    max_attempts: int = 3
    max_batch: int = -1

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
