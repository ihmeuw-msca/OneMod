from typing import Optional

from pydantic import BaseModel


class RoverConfiguration(BaseModel):

    max_attempts: int = 1
    groupby: list[str] = []
    model_type: str
    obs: str
    cov_fixed: list[str] = []
    cov_exploring: list[str] = []
    weights: str
    holdouts: list[str] = []
    fit_args: dict = {}

class RegmodSmoothConfiguration(BaseModel):

    pass


class WeaveConfiguration(BaseModel):
    pass


class SwimrConfiguration(BaseModel):
    pass


class EnsembleConfiguration(BaseModel):
    pass


class ParentConfiguration(BaseModel):
    input_path: str
    col_id: list[str]
    col_obs: str
    col_pred: str
    col_holdout: list[str]
    col_test: str

    rover_covsel: Optional[RoverConfiguration] = None
    regmod_smooth: Optional[RegmodSmoothConfiguration] = None
    weave: Optional[WeaveConfiguration] = None
    swimr: Optional[SwimrConfiguration] = None
    ensemble: Optional[EnsembleConfiguration] = None


settings_file= pathlib

def load_settings(settings_file: str) -> dict:
    yaml.load(settings_file)
    return ParentConfiguration(**yaml.load(settings_file)