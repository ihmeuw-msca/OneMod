"""Preprocessing stage settings."""

from pathlib import Path

from onemod.config import StageConfig


class PreprocessingConfig(StageConfig):
    """Preprocessing stage settings."""

    data: Path
