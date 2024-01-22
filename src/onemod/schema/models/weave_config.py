from onemod.schema.models.base import ParametrizedBaseModel
from typing import Union

class WeaveDimension(ParametrizedBaseModel):
    name: str
    coordinates: str
    kernel: str
    radius: Union[float,list] = 0.0
    exponent: Union[float,list] = 0.0
    coordinates: str | list[str] | None = None

class WeaveModel(ParametrizedBaseModel):
    max_batch: int = 5000
    groupby: list[str] = []
    dimensions: dict[str, WeaveDimension] = {}

class WeaveConfiguration(ParametrizedBaseModel):
    models: dict[str, WeaveModel] | None = None

    def inherit(self):
        super().inherit(keys=['max_batch', 'model_type'])
