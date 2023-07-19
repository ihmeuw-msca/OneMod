"""Run rover model."""
from pathlib import Path
import shutil
from typing import Union
import warnings

import fire
from modrover.info import ModelEval, ModelSpecs, RoverSpecs, SynthSpecs
from modrover.main import Rover
from modrover.modelhub import ModelHub
from modrover.synthesizer import synthesize
import numpy as np
import pandas as pd
from scipy.stats import norm

from onemod.utils import as_list, get_rover_input, load_settings, Subsets


# regmod model parameters
param_dict = {
    "binomial": ["p"],
    "poisson": ["lam"],
    "tobit": ["mu", "sigma"],
}


def get_residual(row: pd.Series, model_type: str, col_obs: str, inv_link: str) -> float:
    """Get residual."""
    if model_type == "binomial" and inv_link == "expit":
        return (row[col_obs] - row["p"]) / (row["p"] * (1 - row["p"]))
    if model_type == "poisson" and inv_link == "exp":
        return row[col_obs] / row["lam"] - 1
    if model_type == "tobit" and inv_link == "exp":
        if row[col_obs] > 0:
            return row[col_obs] / row["mu"] - 1
        w = row["mu"] / row["sigma"]
        term = w * np.imag(norm.logcdf(-w + 1e-6j)) / (1e-6)
        return -1 / (1 - w**2 + term)
    raise ValueError("Unsupported model_type and inv_link pair")


def get_residual_se(
    row: pd.Series, model_type: str, col_obs: str, inv_link: str
) -> float:
    """Get residual standard error."""
    if model_type == "binomial" and inv_link == "expit":
        return 1 / np.sqrt(row["p"] * (1 - row["p"]))
    if model_type == "poisson" and inv_link == "exp":
        return 1 / np.sqrt(row["lam"])
    if model_type == "tobit" and inv_link == "exp":
        if row[col_obs] > 0:
            return row["sigma"] / row["mu"]
        w = row["mu"] / row["sigma"]
        term = w * np.imag(norm.logcdf(-w + 1e-6j)) / (1e-6)
        return np.sqrt(1 / (term * (1 - w**2 + term)))
    raise ValueError("Unsupported model_type and inv_link pair")


def rover_model(experiment_dir: Union[Path, str], submodel_id: str) -> None:
    """Run rover model by submodel ID."""
    experiment_dir = Path(experiment_dir)
    rover_dir = experiment_dir / "results" / "rover"
    settings = load_settings(experiment_dir / "config" / "settings.yml")
    subsets = Subsets(
        "rover", settings["rover"], subsets=pd.read_csv(rover_dir / "subsets.csv")
    )

    # Load data and filter by subset
    subset_id = int(submodel_id[6:])
    df_input = subsets.filter_subset(get_rover_input(settings), subset_id)
    df_train = df_input[df_input[settings["col_test"]] == 0]
    df_train.to_parquet(rover_dir / "data" / f"{submodel_id}.parquet")

    # Create rover objects
    for col in ["offset", "weights"]:  # default column names
        if f"col_{col}" not in settings["rover"]:
            settings["rover"][f"col_{col}"] = col
    model_specs = ModelSpecs(
        col_id=as_list(settings["col_id"]),
        col_obs=settings["col_obs"],
        col_fixed_covs=settings["rover"]["col_fixed_covs"],
        col_covs=settings["rover"]["col_covs"],
        col_holdout=as_list(settings["col_holdout"]),
        col_offset=settings["rover"]["col_offset"],
        col_weights=settings["rover"]["col_weights"],
        inv_link=settings["rover"]["inv_link"],
        model_type=settings["rover"]["model_type"],
    )
    rover = Rover(
        specs=RoverSpecs(strategy_names=settings["rover"]["strategy_names"]),
        modelhub=ModelHub(
            input_path=rover_dir / "data" / f"{submodel_id}.parquet",
            output_dir=rover_dir / "submodels" / submodel_id,
            model_specs=model_specs,
            model_eval=ModelEval(metric=settings["rover"]["eval_metric"]),
        ),
    )

    # Run rover model
    rover.explore(verbose=1)

    # Remove any failed submodels
    for submodel_dir in [
        child
        for child in (rover_dir / "submodels" / submodel_id).iterdir()
        if child.is_dir()
    ]:
        if "performance.yaml" not in [child.name for child in submodel_dir.iterdir()]:
            msg = f"Submodel '{submodel_dir.name}' failed. Removing directory."
            warnings.warn(msg)
            shutil.rmtree(submodel_dir)

    # Get ensemble model
    cov_ids = tuple(np.arange(1, len(settings["rover"]["col_covs"]) + 1))
    df_synth = synthesize(
        df=rover.collect(), synth_specs=SynthSpecs(), required_covs=model_specs.all_covs
    )
    df_coefs = df_synth[["cov_name", "mean", "sd"]]
    model = rover.modelhub._get_model(cov_ids, df_coefs=df_coefs)

    # Get predictions
    if (
        settings["rover"]["model_type"] == "tobit"
        and settings["rover"]["inv_link"] == "exp"
    ):
        df_input["log_sigma"] = 1
    df_pred = model.predict(df_input)[
        as_list(settings["col_id"])
        + [settings["col_obs"]]
        + param_dict[settings["rover"]["model_type"]]
    ]
    df_pred[settings["col_pred"]] = df_pred[rover.modelhub.specs.model_param_name]

    # Get residuals
    df_pred.reset_index(drop=True, inplace=True)
    df_pred["residual"] = df_pred.apply(
        lambda row: get_residual(
            row,
            settings["rover"]["model_type"],
            settings["col_obs"],
            settings["rover"]["inv_link"],
        ),
        axis=1,
    )
    df_pred["residual_se"] = df_pred.apply(
        lambda row: get_residual_se(
            row,
            settings["rover"]["model_type"],
            settings["col_obs"],
            settings["rover"]["inv_link"],
        ),
        axis=1,
    )

    # Save predictions
    df_pred.drop(columns=settings["col_obs"], inplace=True)
    df_pred.to_parquet(rover_dir / "submodels" / submodel_id / "predictions.parquet")


def main() -> None:
    fire.Fire(rover_model)
