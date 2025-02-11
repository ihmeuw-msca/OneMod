"""Kreg stage.

TODO: Implement stage

"""

from typing import Any

from onemod.config import KregConfig
from onemod.stage import Stage


class KregStage(Stage):
    """Kreg stage."""

    config: KregConfig
    _required_input: list[str] = ["data.parquet"]
    _optional_input: list[str] = ["offset.parquet", "priors.pkl"]
    _output: list[str] = ["predictions.parquet", "model.pkl"]

    def _run(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Run kreg submodel."""
        print(
            f"running {self.name} submodel: subset {subset}, paramset {paramset}"
        )
        self._fit(subset, paramset)
        self._predict(subset, paramset)

    def _fit(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Fit kreg submodel."""
        print(
            f"fitting {self.name} submodel: subset {subset}, paramset {paramset}"
        )

    def _predict(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        "Create kreg submodel predictions."
        print(
            f"predicting for {self.name} submodel: subset {subset}, paramset {paramset}"
        )

    def collect(self) -> None:
        """Collect kreg submodel results."""
        print(f"collecting {self.name} submodel results")
