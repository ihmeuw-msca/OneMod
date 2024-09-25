"""Rover stage settings."""

from onemod.config import ModelConfig


class RoverConfig(ModelConfig):
    """Rover stage settings."""

    cov_exploring: set[str]
    cov_fixed: set[str] = {"intercept"}
