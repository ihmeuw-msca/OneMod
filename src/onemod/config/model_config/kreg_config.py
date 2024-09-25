"""Kreg stage settings."""

from pathlib import Path

from onemod.config import ModelConfig


class KregConfig(ModelConfig):
    """Kreg stage settings."""

    offset: Path | None = None
