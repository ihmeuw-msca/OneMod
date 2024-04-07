from onemod.schema.base import Config, StageConfig


class DimensionInit(Config):
    """Weave dimension creation configuration."""

    name: str
    coordinates: str | list[str] | None = None
    kernel: str = "identity"
    distance: str | None = None
    radius: float | list[float] = [0.0]
    exponent: float | list[float] = [0.0]
    version: str | None = None
    distance_dict: str | list[str] | None = None


class WeaveConfig(StageConfig):
    """Configuration for the WeAve stage."""

    max_batch: int = 5000
    dimensions: dict[str, DimensionInit] = {}
