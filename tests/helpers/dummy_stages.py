from onemod.config import KregConfig, ModelConfig, PreprocessingConfig, RoverConfig, SpxmodConfig
from onemod.stage import ModelStage, Stage


class CustomConfig(ModelConfig):
    """Custom stage config."""

    custom_param: int | set[int] = 1
    _crossable_params: set[str] = {"custom_param"}


class DummyCustomStage(ModelStage):
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


class DummyKregStage(ModelStage):
    """Kreg stage."""

    config: KregConfig
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {"offset.parquet", "priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}

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


class DummyPreprocessingStage(Stage):
    """Preprocessing stage."""

    config: PreprocessingConfig
    _skip: set[str] = {"predict"}
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {
        "age_metadata.parquet",
        "location_metadata.parquet",
    }
    _output: set[str] = {"data.parquet"}

    def run(self) -> None:
        """Run preprocessing stage."""
        print(f"running {self.name}")


class DummyRoverStage(ModelStage):
    """Rover stage."""

    config: RoverConfig
    _skip: set[str] = {"predict"}
    _required_input: set[str] = {"data.parquet"}
    _output: set[str] = {"selected_covs.csv"}

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run rover submodel."""
        print(
            f"running {self.name} submodel: subset {subset_id}, param set {param_id}"
        )
        self.fit(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit rover submodel."""
        print(
            f"fitting {self.name} submodel: subset {subset_id}, param set {param_id}"
        )

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        "predict() is not implemented for RoverStage."
        pass

    def collect(self) -> None:
        """Collect rover submodel results."""
        print(f"collecting {self.name} submodel results")


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

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run spxmod submodel."""
        print(
            f"running {self.name} submodel: subset {subset_id}, param set {param_id}"
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
