from onemod.schema.models.base import ParametrizedBaseModel

class RegmodSmoothConfiguration(ParametrizedBaseModel):

    model_type: str = ""
    dims: list[dict] = []
    var_groups: list[dict] = []
    weights: str
    fit_args: dict = {}
    inv_link: str
    coef_bounds: dict[str, list[float]] = {}
    lam: float = 0.0

    def inherit(self):
        super().inherit(keys=['model_type'])
