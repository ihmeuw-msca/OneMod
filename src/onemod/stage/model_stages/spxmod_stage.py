"""Spxmod stage."""

import fire

from onemod.config.model_config import SpxmodConfig
from onemod.stage import ModelStage


class SpxmodStage(ModelStage):
    """Spxmod stage."""

    config: SpxmodConfig = SpxmodConfig()

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run spxmod submodel."""
        print(
            f"running spxmod submodel: subset {subset_id}, param set {param_id}"
        )
        self.fit(subset_id, param_id)
        self.predict(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit spxmod submodel."""
        print(
            f"fitting {self.name} submodel: subset {subset_id}, param set {param_id}"
        )

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        "Create spxmod submodel predictions."
        print(
            f"predicting for {self.name} submodel: subset {subset_id}, param set {param_id}"
        )

    def collect(self) -> None:
        """Collect spxmod submodel results."""
        print(f"collecting {self.name} submodel results")


if __name__ == "__main__":
    fire.Fire(SpxmodStage.evaluate)
