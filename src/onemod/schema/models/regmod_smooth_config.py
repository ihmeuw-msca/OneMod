from pydantic import Field

from onemod.schema.models.base import ParametrizedBaseModel


class RegmodModelInit(ParametrizedBaseModel):
    model_type: str = ""
    dims: list[dict] = []
    var_groups: list[dict] = []
    weights: str

    coef_bounds: dict[str, list[float]] = {}
    lam: float = Field(0.0, alias="lambda")


class RegmodSmoothConfiguration(ParametrizedBaseModel):
    max_attempts: int = 3
    groupby: list[str] = []
    fit_args: dict = {}

    Model: RegmodModelInit

    def inherit(self) -> None:
        super().inherit(keys=["model_type", "groupby", "max_attempts", "max_batch"])
