"""Kreg stage."""

import fire

from onemod.config.model_config import KregConfig
from onemod.stage import ModelStage


class KregStage(ModelStage):
    """Kreg stage."""

    config: KregConfig = KregConfig()

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run kreg submodel."""
        print(
            f"running {self.name} submodel: subset {subset_id}, param set {param_id}"
        )
        self.fit(subset_id, param_id)
        self.predict(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit kreg submodel."""
        print(
            f"fitting {self.name} submodel: subset {subset_id}, param set {param_id}"
        )

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        "Create kreg submodel predictions."
        print(
            f"predicting for {self.name} submodel: subset {subset_id}, param set {param_id}"
        )

    def collect(self) -> None:
        """Collect kreg submodel results."""
        print(f"collecting {self.name} submodel results")


if __name__ == "__main__":
    fire.Fire(KregStage.evaluate)
