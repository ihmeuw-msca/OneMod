"""Preprocessing stage."""

import fire

from onemod.config import PreprocessingConfig
from onemod.stage import Stage


class PreprocessingStage(Stage):
    """Preprocessing stage."""

    config: PreprocessingConfig
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {
        "age_metadata.parquet",
        "location_metadata.parquet",
    }
    _output: set[str] = {"data.parquet"}

    def run(self) -> None:
        """Run preprocessing stage."""
        print(f"running {self.name}")


if __name__ == "__main__":
    fire.Fire(PreprocessingStage.evaluate)
