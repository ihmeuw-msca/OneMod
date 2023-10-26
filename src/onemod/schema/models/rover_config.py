from onemod.schema.models.base import ParametrizedBaseModel

class RoverConfiguration(ParametrizedBaseModel):

    groupby: list[str] = []
    model_type: str = ""
    cov_fixed: list[str] = []
    cov_exploring: list[str] = []
    weights: str
    holdouts: list[str] = []
    fit_args: dict = {}

    def inherit(self):
        super().inherit(keys=['model_type'])
