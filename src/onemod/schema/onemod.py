from typing import Any, Literal

from onemod.schema.base import Config
from onemod.schema.stages import (
    EnsembleConfig,
    RegmodSmoothConfig,
    RoverCovselConfig,
    WeaveConfig,
)


class OneModConfig(Config):
    """OneMod configuration class. It holds global information about the run. And
    each stage configuration is stored in a separate class.

    """

    input_path: str
    ids: list[str]
    obs: str
    mtype: Literal["binomial", "gaussian", "poisson"]
    weights: str
    pred: str
    holdouts: list[str]
    test: str
    id_subsets: dict[str, list[Any]] = {}

    rover_covsel: RoverCovselConfig | None = None
    regmod_smooth: RegmodSmoothConfig | None = None
    weave: dict[str, WeaveConfig] | None = None
    ensemble: EnsembleConfig | None = None
