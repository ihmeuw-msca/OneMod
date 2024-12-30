from pydantic import Field

from onemod.config import KregConfig, RoverConfig, SpxmodConfig, StageConfig
from onemod.stage import ModelStage, Stage


class CustomConfig(StageConfig):
    """Custom stage config."""

    custom_param: int | set[int] = 1
    _crossable_params: set[str] = {"custom_param"}


class DummyCustomStage(ModelStage):
    """Custom stage."""

    config: CustomConfig = CustomConfig()  # type: ignore
    _required_input: set[str] = {"observations.parquet", "predictions.parquet"}
    _collect_after: set[str] = {"run", "predict"}

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run custom submodel."""
        self.log.append(
            f"run: name={self.name}, subset={subset_id}, param={param_id}"
        )
        self.fit(subset_id, param_id)
        self.predict(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit custom submodel."""
        self.log.append(
            f"fit: name={self.name}, subset={subset_id}, param={param_id}"
        )

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Create custom submodel predictions."""
        self.log.append(
            f"predict: name={self.name}, subset={subset_id}, param={param_id}"
        )

    def collect(self) -> None:
        """Collect custom submodel results."""
        self.log.append(f"collect: name={self.name}")

    # Dummy-specific methods
    def get_log(self) -> list[str]:
        """Retrieve the internal log."""
        return self.log


class DummyKregStage(ModelStage):
    """Kreg stage."""

    config: KregConfig
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {"offset.parquet", "priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}
    _collect_after: set[str] = {"run", "predict"}

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run kreg submodel."""
        self.log.append(
            f"run: name={self.name}, subset={subset_id}, param={param_id}"
        )
        self.fit(subset_id, param_id)
        self.predict(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit kreg submodel."""
        self.log.append(
            f"fit: name={self.name}, subset={subset_id}, param={param_id}"
        )

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        "Create kreg submodel predictions."
        self.log.append(
            f"predict: name={self.name}, subset={subset_id}, param={param_id}"
        )

    def collect(self) -> None:
        """Collect kreg submodel results."""
        self.log.append(f"collect: name={self.name}")

    # Dummy-specific methods
    def get_log(self) -> list[str]:
        """Retrieve the internal log."""
        return self.log


class DummyPreprocessingStage(Stage):
    """Preprocessing stage."""

    config: StageConfig
    _skip: set[str] = {"predict"}
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {
        "age_metadata.parquet",
        "location_metadata.parquet",
    }
    _output: set[str] = {"data.parquet"}

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def run(self) -> None:
        """Run preprocessing stage."""
        self.log.append(f"run: name={self.name}")

    # Dummy-specific methods
    def get_log(self) -> list[str]:
        """Retrieve the internal log."""
        return self.log


class DummyRoverStage(ModelStage):
    """Rover stage."""

    config: RoverConfig
    _skip: set[str] = {"predict"}
    _required_input: set[str] = {"data.parquet"}
    _output: set[str] = {"selected_covs.csv"}
    _collect_after: set[str] = {"run", "fit"}

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run rover submodel."""
        self.log.append(
            f"run: name={self.name}, subset={subset_id}, param={param_id}"
        )
        self.fit(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit rover submodel."""
        self.log.append(
            f"fit: name={self.name}, subset={subset_id}, param={param_id}"
        )

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """predict() is not implemented for RoverStage."""
        pass

    def collect(self) -> None:
        """Collect rover submodel results."""
        self.log.append(f"collect: name={self.name}")

    # Dummy-specific methods
    def get_log(self) -> list[str]:
        """Retrieve the internal log."""
        return self.log


class DummySpxmodStage(ModelStage):
    """Spxmod stage."""

    config: SpxmodConfig
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {
        "selected_covs.csv",
        "offset.parquet",
        "priors.pkl",
    }
    _output: set[str] = {"predictions.parquet", "model.pkl"}
    _collect_after: set[str] = {"run", "predict"}

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run spxmod submodel."""
        self.log.append(
            f"run: name={self.name}, subset={subset_id}, param={param_id}"
        )
        self.fit(subset_id, param_id)
        self.predict(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit spxmod submodel."""
        self.log.append(
            f"fit: name={self.name}, subset={subset_id}, param={param_id}"
        )

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        "Create spxmod submodel predictions."
        self.log.append(
            f"predict: name={self.name}, subset={subset_id}, param={param_id}"
        )

    def collect(self) -> None:
        """Collect spxmod submodel results."""
        self.log.append(f"collect: name={self.name}")

    # Dummy-specific methods
    def get_log(self) -> list[str]:
        """Retrieve the internal log."""
        return self.log


class MultiplyByTwoStage(ModelStage):
    """Stage that multiplies the value column by 2."""

    config: StageConfig
    _skip: set[str] = {"predict"}
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {
        "age_metadata.parquet",
        "location_metadata.parquet",
    }
    _output: set[str] = {"data.parquet"}

    def run(self, subset_id: int, *args, **kwargs) -> None:
        """Run MultiplyByTwoStage."""
        df = self.get_stage_subset(subset_id)
        df["value"] = df["value"] * 2
        self.dataif.dump(df, "data.parquet", key="output")

    def fit(self) -> None:
        """Fit MultiplyByTwoStage."""
        pass

    def predict(self) -> None:
        """Predict MultiplyByTwoStage."""
        pass

    def collect(self) -> None:
        """Collect MultiplyByTwoStage."""
        pass


def assert_stage_logs(
    stage: DummyCustomStage
    | DummyKregStage
    | DummyRoverStage
    | DummySpxmodStage,
    methods: list[str] | None = None,
    subset_ids: list[int] | None = None,
    param_ids: list[int] | None = None,
):
    """Assert that the expected methods were logged for a given stage."""
    log = stage.get_log()
    if methods:
        for method in methods:
            if subset_ids:
                for subset_id in subset_ids:
                    if param_ids:
                        for param_id in param_ids:
                            assert (
                                f"{method}: name={stage.name}, subset={subset_id}, param={param_id}"
                                in log
                            )
                    else:
                        assert (
                            f"{method}: name={stage.name}, subset={subset_id}, param=None"
                            in log
                        )
            if method in stage._collect_after:
                assert f"collect: name={stage.name}" in log
