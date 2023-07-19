"""Collect onemod stage submodel results."""
from pathlib import Path
from typing import Union

import fire
import pandas as pd

from onemod.utils import (
    as_list,
    get_rover_submodels,
    get_swimr_submodels,
    get_weave_submodels,
    load_settings,
)


def collect_rover_results(experiment_dir: Union[Path, str]) -> None:
    """Collect rover submodel results."""
    experiment_dir = Path(experiment_dir)
    rover_dir = experiment_dir / "results" / "rover"
    submodel_ids = get_rover_submodels(experiment_dir)
    pd.concat(
        [
            pd.read_parquet(
                rover_dir / "submodels" / submodel_id / "predictions.parquet"
            )
            for submodel_id in submodel_ids
        ],
        ignore_index=True,
    ).to_parquet(rover_dir / "predictions.parquet")


def collect_swimr_results(experiment_dir: Union[Path, str]) -> None:
    """Collect swimr submodel results."""
    experiment_dir = Path(experiment_dir)
    swimr_dir = experiment_dir / "results" / "swimr"
    settings = load_settings(experiment_dir / "config" / "settings.yml")
    submodel_ids = get_swimr_submodels(experiment_dir)
    for holdout_id in as_list(settings["col_holdout"]) + ["full"]:
        df_list = []
        for submodel_id in [
            submodel_id
            for submodel_id in submodel_ids
            if submodel_id.split("_")[3] == holdout_id
        ]:
            df_list.append(
                pd.read_parquet(
                    swimr_dir / "submodels" / submodel_id / "predictions.parquet"
                ).astype({"param_id": str})
            )
        df_pred = pd.pivot(
            data=pd.concat(df_list, ignore_index=True),
            index=settings["col_id"],
            columns=["model_id", "param_id"],
            values=["residual", settings["col_pred"]],
        )
        if holdout_id == "full":
            df_pred.to_parquet(swimr_dir / "predictions.parquet")
        else:
            df_pred.to_parquet(swimr_dir / f"predictions_{holdout_id}.parquet")


def collect_weave_results(experiment_dir: Union[Path, str]) -> None:
    """Collect weave submodel results."""
    experiment_dir = Path(experiment_dir)
    weave_dir = experiment_dir / "results" / "weave"
    settings = load_settings(experiment_dir / "config" / "settings.yml")
    submodel_ids = get_weave_submodels(experiment_dir)
    for holdout_id in as_list(settings["col_holdout"]) + ["full"]:
        df_pred = pd.concat(
            [
                pd.read_parquet(
                    weave_dir / "submodels" / f"{submodel_id}.parquet"
                ).astype({"param_id": str})
                for submodel_id in submodel_ids
                if submodel_id.split("_")[3] == holdout_id
            ],
            ignore_index=True,
        )
        df_pred = pd.pivot(
            data=df_pred,
            index=settings["col_id"],
            columns=["model_id", "param_id"],
            values=["residual", settings["col_pred"]],
        )
        if holdout_id == "full":
            df_pred.to_parquet(weave_dir / "predictions.parquet")
        else:
            df_pred.to_parquet(weave_dir / f"predictions_{holdout_id}.parquet")


def main() -> None:
    fire.Fire(
        {
            "rover": collect_rover_results,
            "swimr": collect_swimr_results,
            "weave": collect_weave_results,
        }
    )
