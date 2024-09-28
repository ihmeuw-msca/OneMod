"""Preprocessing stage settings."""

from pathlib import Path

from onemod.redesign.config import StageConfig


class PreprocessingConfig(StageConfig):
    """Preprocessing stage settings."""

    data: Path
