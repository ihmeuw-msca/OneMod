from onemod.schema.models.base import ParametrizedBaseModel


class WeaveDimension(ParametrizedBaseModel):
    name: str
    coordinates: str
    kernel: str
    radius: float = 0.0
    exponent: float = 0.0
    coordinates: str | list[str] | None = None

class WeaveModel(ParametrizedBaseModel):
    max_batch: int = 5000
    groupby: list[str] = []
    dimensions: dict[str, WeaveDimension] = {}

class WeaveConfiguration(ParametrizedBaseModel):

    models: dict[str, WeaveModel] | None = None

    def inherit(self):
        super().inherit(keys=['max_batch', 'model_type'])
