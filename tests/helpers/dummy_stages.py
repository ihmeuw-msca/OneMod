from typing import Any

from pandas import DataFrame
from pydantic import Field

from onemod.config import KregConfig, RoverConfig, SpxmodConfig, StageConfig
from onemod.stage import Stage


class CustomConfig(StageConfig):
    """Custom stage config."""

    custom_param: int | list[int] = 1


class DummyCustomStage(Stage):
    """Custom stage."""

    config: CustomConfig = CustomConfig()  # type: ignore
    _required_input: list[str] = ["observations.parquet", "predictions.parquet"]
    _collect_after: list[str] = ["run", "predict"]

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def _run(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Run custom submodel."""
        self.log.append(
            f"run: name={self.name}, subset={subset}, paramset={paramset}"
        )
        self._fit(subset, paramset)
        self._predict(subset, paramset)

    def _fit(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Fit custom submodel."""
        self.log.append(
            f"fit: name={self.name}, subset={subset}, paramset={paramset}"
        )

    def _predict(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Create custom submodel predictions."""
        self.log.append(
            f"predict: name={self.name}, subset={subset}, paramset={paramset}"
        )

    def collect(self) -> None:
        """Collect custom submodel results."""
        self.log.append(f"collect: name={self.name}")

    # Dummy-specific methods
    def get_log(self) -> list[str]:
        """Retrieve the internal log."""
        return self.log


class DummyKregStage(Stage):
    """Kreg stage."""

    config: KregConfig
    _required_input: list[str] = ["data.parquet"]
    _optional_input: list[str] = ["offset.parquet", "priors.pkl"]
    _output: list[str] = ["predictions.parquet", "model.pkl"]
    _collect_after: list[str] = ["run", "predict"]

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def _run(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Run kreg submodel."""
        self.log.append(
            f"run: name={self.name}, subset={subset}, paramset={paramset}"
        )
        self._fit(subset, paramset)
        self._predict(subset, paramset)

    def _fit(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Fit kreg submodel."""
        self.log.append(
            f"fit: name={self.name}, subset={subset}, paramset={paramset}"
        )

    def _predict(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        "Create kreg submodel predictions."
        self.log.append(
            f"predict: name={self.name}, subset={subset}, paramset={paramset}"
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
    _skip: list[str] = ["predict"]
    _required_input: list[str] = ["data.parquet"]
    _optional_input: list[str] = [
        "age_metadata.parquet",
        "location_metadata.parquet",
    ]
    _output: list[str] = ["data.parquet"]

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def _run(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Run preprocessing stage."""
        self.log.append(
            f"run: name={self.name}, subset={subset}, paramset={paramset}"
        )

    def _fit(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        self._run(subset, paramset)

    # Dummy-specific methods
    def get_log(self) -> list[str]:
        """Retrieve the internal log."""
        return self.log


class DummyRoverStage(Stage):
    """Rover stage."""

    config: RoverConfig
    _skip: list[str] = ["predict"]
    _required_input: list[str] = ["data.parquet"]
    _output: list[str] = ["selected_covs.csv"]
    _collect_after: list[str] = ["run", "fit"]

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def _run(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Run rover submodel."""
        self.log.append(
            f"run: name={self.name}, subset={subset}, paramset={paramset}"
        )
        self._fit(subset, paramset)

    def _fit(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Fit rover submodel."""
        self.log.append(
            f"fit: name={self.name}, subset={subset}, paramset={paramset}"
        )

    def _predict(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
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


class DummySpxmodStage(Stage):
    """Spxmod stage."""

    config: SpxmodConfig
    _required_input: list[str] = ["data.parquet"]
    _optional_input: list[str] = [
        "selected_covs.csv",
        "offset.parquet",
        "priors.pkl",
    ]
    _output: list[str] = ["predictions.parquet", "model.pkl"]
    _collect_after: list[str] = ["run", "predict"]

    # Dummy-specific attributes
    log: list[str] = Field(default_factory=list, exclude=True)

    def _run(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Run spxmod submodel."""
        self.log.append(
            f"run: name={self.name}, subset={subset}, paramset={paramset}"
        )
        self._fit(subset, paramset)
        self._predict(subset, paramset)

    def _fit(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        """Fit spxmod submodel."""
        self.log.append(
            f"fit: name={self.name}, subset={subset}, paramset={paramset}"
        )

    def _predict(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        "Create spxmod submodel predictions."
        self.log.append(
            f"predict: name={self.name}, subset={subset}, paramset={paramset}"
        )

    def collect(self) -> None:
        """Collect spxmod submodel results."""
        self.log.append(f"collect: name={self.name}")

    # Dummy-specific methods
    def get_log(self) -> list[str]:
        """Retrieve the internal log."""
        return self.log


class MultiplyByTwoStage(Stage):
    """Stage that multiplies the value column by 2."""

    config: StageConfig
    _skip: list[str] = ["predict"]
    _required_input: list[str] = ["data.parquet"]
    _optional_input: list[str] = [
        "age_metadata.parquet",
        "location_metadata.parquet",
    ]
    _output: list[str] = ["data.parquet"]

    def _run(self, subset: dict[str, Any], *args, **kwargs) -> None:
        """Run MultiplyByTwoStage."""
        df = self.dataif.load(key="data", subset=subset)
        df["value"] = df["value"] * 2
        self.dataif.dump(df, "data.parquet", key="output")

    def _fit(self) -> None:
        """Fit MultiplyByTwoStage."""
        pass

    def _predict(self) -> None:
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
    subsets: DataFrame | None = None,
    paramsets: DataFrame | None = None,
):
    """Assert that the expected methods were logged for a given stage."""
    log = stage.get_log()
    if methods:
        for method in methods:
            if subsets is not None:
                for subset in subsets.to_dict(orient="records"):
                    if paramsets is not None:
                        for paramset in paramsets.to_dict(orient="records"):
                            assert (
                                f"{method}: name={stage.name}, subset={subset}, paramset={paramset}"
                                in log
                            )
                    else:
                        assert (
                            f"{method}: name={stage.name}, subset={subset}, paramset=None"
                            in log
                        )
                else:
                    assert f"{method}: name={stage.name}, subset=None, paramset=None"
            else:
                assert f"{method}: name={stage.name}"
            if method in stage._collect_after:
                assert f"collect: name={stage.name}" in log
