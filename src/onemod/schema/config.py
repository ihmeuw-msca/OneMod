from typing import Optional

from pydantic import BaseModel, field_validator

from modrover.globals import model_type_dict


class RoverConfiguration(BaseModel):

    groupby: list[str] = []
    model_type: str
    obs: str
    cov_fixed: list[str] = []
    cov_exploring: list[str] = []
    weights: str
    holdouts: list[str] = []
    fit_args: dict = {}

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


class RegmodSmoothConfiguration(BaseModel):

    model_type: str
    obs: str
    dims: list[dict] = []
    var_groups: list[dict] = []
    weights: str
    fit_args: dict = {}
    coef_bounds: list[dict] = []

    @field_validator("model_type")
    @classmethod
    def valid_model_type(cls, model_type: str) -> str:
        assert model_type in model_type_dict, \
            f"model_type must be one of {model_type_dict.keys()}"
        return model_type


class WeaveConfiguration(BaseModel):
    pass   # TODO


class SwimrConfiguration(BaseModel):
    pass  # TODO


class EnsembleConfiguration(BaseModel):
    pass  # TODO


class ParentConfiguration(BaseModel, extra='allow'):

    input_path: str
    col_id: list[str]
    col_obs: str
    col_pred: str
    col_holdout: list[str]
    col_test: str
    max_attempts: int = 3

    rover_covsel: Optional[RoverConfiguration] = None
    regmod_smooth: Optional[RegmodSmoothConfiguration] = None
    weave: Optional[WeaveConfiguration] = None
    swimr: Optional[SwimrConfiguration] = None
    ensemble: Optional[EnsembleConfiguration] = None

    @property
    def extra_fields(self) -> set[str]:
        return set(self.__dict__) - set(self.model_fields)

    def validate_against_dataset(self, dataset: pd.DataFrame):
        pass # TODO
