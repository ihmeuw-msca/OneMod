from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.functional_validators import field_validator

from modrover.globals import model_type_dict

CFG_ALIASES = {
    "mtype": "model_type",
}


class CFG(BaseModel):
    """An extension of BaseModel that supports __getitem__ and is configured."""

    model_config = ConfigDict(extra="allow", frozen=False, validate_assignment=True)

    @property
    def extra_fields(self) -> set[str]:
        return set(self.__dict__) - set(self.model_fields)

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields

    def __getitem__(self, key: str) -> Any:
        if key in self:
            return getattr(self, key)
        raise KeyError(f"{key} is not in the {self.model_fields=:}")

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Overwrite the parent function to change name in the dictionary
        accroding to CFG_ALIASES.
        """
        cfg = super().model_dump(*args, **kwargs)
        keys_to_rename = [key for key in cfg.keys() if key in CFG_ALIASES]
        for key in keys_to_rename:
            cfg[CFG_ALIASES[key]] = cfg.pop(key)
        return cfg


class StageCFG(CFG):
    groupby: list[str] = []
    max_attempts: int = 1
    max_batch: int = -1


class RoverCFG(CFG):
    mtype: str
    cov_fixed: list[str] = []
    cov_exploring: list[str] = []
    weights: str
    holdouts: list[str] = []

    @field_validator("mtype")
    @classmethod
    def valid_model_type(cls, mtype: str) -> str:
        assert (
            mtype in model_type_dict
        ), f"{mtype=:} is not one of {model_type_dict.keys()}"
        return mtype


class RoverCovSelCFG(StageCFG):
    rover: RoverCFG
    rover_fit: dict = {}

    @field_validator("rover_fit")
    @classmethod
    def valid_rover_fit(cls, rover_fit: dict):
        # TODO: Necessary or not to import and validate?
        # Could import Rover.fit and inspect the args
        return rover_fit


class ModelCFG(CFG):
    mtype: str
    dims: list[dict] = []
    var_groups: list[dict] = []
    weights: str
    coef_bounds: dict[str, list[float]] = {}
    lam: float = 0.0

    @field_validator("mtype")
    @classmethod
    def valid_model_type(cls, mtype: str) -> str:
        assert (
            mtype in model_type_dict
        ), f"{mtype=:} is not one of {model_type_dict.keys()}"
        return mtype

    def to_args(self, obs: str, covs: list[str]) -> dict[str, Any]:
        args = self.model_dump()

        var_groups = args["var_groups"]
        coef_bounds = args.pop("coef_bounds")
        lam = args.pop("lam")

        var_group_keys = [
            (var_group["col"], var_group.get("dim")) for var_group in var_groups
        ]

        for cov in covs:
            if (cov, "age_mid") not in var_group_keys:
                var_groups.append(dict(col=cov, dim="age_mid"))

        for var_group in var_groups:
            cov = var_group["col"]
            if "uprior" not in var_group:
                var_group["uprior"] = tuple(
                    map(float, coef_bounds.get(cov, [-100, 100]))
                )
            if "lam" not in var_group:
                var_group["lam"] = lam

        args["obs"] = obs
        return args


class RegmodSmoothCFG(StageCFG):
    model: ModelCFG
    model_fit: dict = {}

    @field_validator("model_fit")
    @classmethod
    def valid_rover_fit(cls, model_fit: dict):
        # TODO: Necessary or not to import and validate?
        # Could import Model.fit and inspect the args
        return model_fit


class WeaveCFG(StageCFG):
    # TODO
    pass


class SwimrCFG(StageCFG):
    # TODO
    pass


class EnsembleCFG(StageCFG):
    # TODO
    pass


class OneModCFG(CFG):
    input_path: str
    col_id: list[str]
    col_obs: str
    col_pred: str
    col_holdout: list[str]
    col_test: str
    col_sigma: str = ""

    rover_covsel: RoverCovSelCFG | None = None
    regmod_smooth: RegmodSmoothCFG | None = None
    weave: WeaveCFG | None = None
    swimr: SwimrCFG | None = None
    ensemble: EnsembleCFG | None = None
