"""Preprocessing stage."""

import fire

from onemod.config import PreprocessingConfig
from onemod.stage import Stage


class PreprocessingStage(Stage):
    """Preprocessing stage."""

    config: PreprocessingConfig

    def run(self) -> None:
        """Run preprocessing stage."""
        print(f"running {self.name}")


if __name__ == "__main__":
    fire.Fire(PreprocessingStage.evaluate)
