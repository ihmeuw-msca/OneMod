from onemod.schema.models.base import ParametrizedBaseModel
from pydantic import Field


class RegmodModelInit(ParametrizedBaseModel):
    mtype: str = Field("", alias="model_type")
    dims: list[dict] = []
    var_groups: list[dict] = []
    weights: str

    coef_bounds: dict[str, list[float]] = {}
    lam: float = 0.0

class RegmodSmoothConfiguration(ParametrizedBaseModel):

    max_attempts: int = 3
    groupby: list[str] = []
    fit_args: dict = {}

    model: RegmodModelInit

    def inherit(self):
        super().inherit(keys=['mtype', 'groupby', 'max_attempts', 'max_batch'])
