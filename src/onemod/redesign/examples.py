"""Example stages."""

from pathlib import Path

from onemod.redesign.config import ModelConfig, StageConfig
from onemod.redesign.stage import ModelStage, Stage


class PreprocessingConfig(StageConfig):
    data: Path


class PreprocessingStage(Stage):
    config: PreprocessingConfig
    _skip_if: set[str] = {"predict"}

    def run(self) -> None:
        print("preprocessing data")


class RoverConfig(ModelConfig):
    cov_exploring: set[str]
    cov_fixed: set[str] = {"intercept"}


class RoverStage(ModelStage):
    config: RoverConfig
    _skip_if: set[str] = {"predict"}

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        self.fit(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        print(f"fitting {self.name} submodel: subset {subset_id}")

    def collect(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        print(f"collecting {self.name} submodel results")


class SpxmodConfig(ModelConfig):
    selected_covs: Path
    offset: Path | None = None


class SpxmodStage(ModelStage):
    config: SpxmodConfig

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        self.fit(subset_id, param_id)
        self.predict(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        print(f"fitting {self.name} submodel: subset {subset_id}")

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        print(f"predicting for {self.name} submodel: {subset_id}")

    def collect(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        print(f"collecting {self.name} submodel results")


class KregConfig(ModelConfig):
    offset: Path
    lam: float | set[float]
    _crossable_params: set[str] = {"lam"}


class KregStage(ModelStage):
    config: KregConfig
    _skip_if: set[str] = {"fit"}

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        self.predict(subset_id, param_id)

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        print(f"predicting for {self.name} submodel: {subset_id}")

    def collect(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        print(f"collecting {self.name} submodel results: {param_id}")
