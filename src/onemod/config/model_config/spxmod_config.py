"""Spxmod stage settings."""

from pathlib import Path

from onemod.config import ModelConfig


class SpxmodConfig(ModelConfig):
    """Spxmod stage settings."""

    selected_covs: Path | None = None
    offset: Path | None = None
