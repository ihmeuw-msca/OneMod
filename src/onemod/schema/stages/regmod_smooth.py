from onemod.schema.base import Config, StageConfig


class ModelInit(Config):
    """Arguments for initialiation of a regmodsm model."""

    dims: list[dict] = []
    var_groups: list[dict] = []
    weights: str

    coef_bounds: dict[str, tuple[float, float]] = {}
    lam: float = 0.0


class RegmodSmoothConfig(StageConfig):
    """RegmodSmooth stage configuration class. Will need the `mtype` form the
    upper level to initialize a regmodsm model.

    """

    model: ModelInit
    model_fit: dict = {}
