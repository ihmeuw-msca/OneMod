"""Rover stage."""

import fire

from onemod.config.model_config import RoverConfig
from onemod.stage import ModelStage


class RoverStage(ModelStage):
    """Rover stage."""

    config: RoverConfig
    _skip_if: set[str] = {"predict"}

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run rover submodel."""
        print(
            f"running rover submodel: subset {subset_id}, param set {param_id}"
        )
        self.fit(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit rover submodel."""
        print(
            f"fitting {self.name} submodel: subset {subset_id}, param set {param_id}"
        )

    def collect(self) -> None:
        """Collect rover submodel results."""
        print(f"collecting {self.name} submodel results")


if __name__ == "__main__":
    fire.Fire(RoverStage.evaluate)
