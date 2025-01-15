"""Example custom stage."""

from onemod.config import StageConfig
from onemod.dtypes import UniqueList
from onemod.stage import ModelStage


class CustomConfig(StageConfig):
    """Custom stage config."""

    custom_param: int | set[int] = 1
    _crossable_params: UniqueList[str] = ["custom_param"]


class CustomStage(ModelStage):
    """Custom stage."""

    config: CustomConfig = CustomConfig()
    _required_input: UniqueList[str] = [
        "observations.parquet",
        "predictions.parquet",
    ]

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
