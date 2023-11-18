from pydantic import Field

from onemod.schema.models.base import ParametrizedBaseModel


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
    regmod_fit: dict = {}

    model: RegmodModelInit

    def inherit(self) -> None:
        super().inherit(keys=['mtype', 'groupby', 'max_attempts', 'max_batch'])
