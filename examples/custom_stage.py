"""Example custom stage."""

import fire

from onemod.config import ModelConfig
from onemod.stage import ModelStage


class CustomConfig(ModelConfig):
    """Custom stage config."""

    example_setting: int = 1


class CustomStage(ModelStage):
    """Custom stage."""

    config: CustomConfig = CustomConfig()
    _required_input: set[str] = {"observations.parquet", "predictions.parquet"}

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run custom submodel."""
        print(
            f"running {self.name} submodel: subset {subset_id}, param set {param_id}"
        )
        self.fit(subset_id, param_id)
        self.predict(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit custom submodel."""
        print(
            f"fitting {self.name} submodel: subset {subset_id}, param set {param_id}"
        )

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Create custom submodel predictions."""
        print(
            f"predicting for {self.name} submodel: subset {subset_id}, param set {param_id}"
        )

    def collect(self) -> None:
        """Collect custom submodel results."""
        print(f"collecting {self.name} submodel results")


if __name__ == "__main__":
    fire.Fire(CustomStage.evaluate)
