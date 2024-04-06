from onemod.schema.base import Config


class WeaveDimension(Config):
    name: str
    kernel: str
    radius: float | list[float] = [0.0]
    exponent: float | list[float] = [0.0]
    coordinates: str | list[str] | None = None


class WeaveModel(Config):
    max_batch: int = 5000
    groupby: list[str] = []
    dimensions: dict[str, WeaveDimension] = {}

    def inherit(self) -> None:
        super().inherit(keys=["groupby", "max_batch"])


class WeaveConfig(Config):
    models: dict[str, WeaveModel] | None = None

    def inherit(self) -> None:
        super().inherit(keys=["max_attempts", "model_type", "max_batch"])
