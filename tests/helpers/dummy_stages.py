from typing import Any

from pandas import DataFrame
from pydantic import Field

from onemod.config import StageConfig
from onemod.stage import Stage


class DummyCustomConfig(StageConfig):
    """Custom stage config."""

    custom_param: int | list[int] = 1


class DummyKregConfig(StageConfig):
    """Kreg config."""

    kreg_model: dict
    kreg_fit: dict = {}
    kreg_uncertainty: dict = {}


class DummyRoverConfig(StageConfig):
    """Rover config."""

    cov_exploring: list[str]
    cov_groupby: list[str]


class DummyCustomStage(Stage):
    """Custom stage."""

    config: DummyCustomConfig = DummyCustomConfig()  # type: ignore
    _collect_after: list[str] = ["run", "predict"]
    _required_input: dict[str, dict[str, Any]] = {
        "observations": {"format": "parquet"},
        "predictions": {"format": "parquet"},
    }

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

    config: DummyKregConfig
    _collect_after: list[str] = ["run", "predict"]
    _required_input: dict[str, dict[str, Any]] = {"data": {"format": "parquet"}}
    _optional_input: dict[str, dict[str, Any]] = {
        "offset": {"format": "parquet"},
        "priors": {"format": "pkl"},
    }
    _output_items: dict[str, dict[str, Any]] = {
        "predictions": {"format": "parquet"},
        "model": {"format": "pkl"},
    }

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
    _required_input: dict[str, dict[str, Any]] = {"data": {"format": "parquet"}}
    _optional_input: dict[str, dict[str, Any]] = {
        "age_metadata": {"format": "parquet"},
        "location_metadata": {"format": "parquet"},
    }
    _output_items: dict[str, dict[str, Any]] = {"data": {"format": "parquet"}}

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

    config: DummyRoverConfig
    _skip: list[str] = ["predict"]
    _collect_after: list[str] = ["run", "fit"]
    _required_input: dict[str, dict[str, Any]] = {"data": {"format": "parquet"}}
    _output_items: dict[str, dict[str, Any]] = {
        "selected_covs": {"format": "csv"}
    }

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


class DummySpxmodConfig(StageConfig):
    """Spxmod config."""

    xmodel: dict


class DummySpxmodStage(Stage):
    """Spxmod stage."""

    config: DummySpxmodConfig
    _collect_after: list[str] = ["run", "predict"]
    _required_input: dict[str, dict[str, Any]] = {"data": {"format": "parquet"}}
    _optional_input: dict[str, dict[str, Any]] = {
        "selected_covs": {"format": "csv"},
        "offset": {"format": "parquet"},
        "priors": {"format": "pkl"},
    }
    _output_items: dict[str, dict[str, Any]] = {
        "predictions": {"format": "parquet"},
        "model": {"format": "pkl"},
    }

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
    _required_input: dict[str, dict[str, Any]] = {"data": {"format": "parquet"}}
    _optional_input: dict[str, dict[str, Any]] = {
        "age_metadata": {"format": "parquet"},
        "location_metadata": {"format": "parquet"},
    }
    _output_items: dict[str, dict[str, Any]] = {"data": {"format": "parquet"}}

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
            if method in stage.collect_after:
                assert f"collect: name={stage.name}" in log
