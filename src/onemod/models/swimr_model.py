"""Run swimr model.

Assumes sex_id in groupby setting (i.e., each model has a unique sex_id)
because sex_id is not included in SWiMR output.

"""
from pathlib import Path
import subprocess
from typing import Union

import fire
import numpy as np
import pandas as pd
import yaml

from onemod.utils import (
    as_list,
    get_prediction,
    get_smoother_input,
    load_settings,
    Subsets,
    SwimrParams,
)


class Quoted(str):
    pass


def quoted_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")


yaml.add_representer(Quoted, quoted_presenter)


def get_str(id_list: Union[list, np.ndarray]) -> str:
    """Format list of IDs as a str.

    Output settings files from pyyaml did not include quotes around
    prediction_submodel_ids or prediction_year_ids, and were read by
    swimr package (in R) incorrectly.

    Solution found here:
    https://stackoverflow.com/questions/38369833/pyyaml-and-using-quotes-for-strings-only

    """
    if isinstance(id_list, np.ndarray):
        id_list = list(id_list)
    return Quoted(str(id_list).translate({ord(ii): None for ii in "[ ]"}))


def create_cascade_hierarchy(df_input: pd.DataFrame, model_settings: dict) -> None:
    """Create cascade hierarchy file."""
    rename = {"locid": "location_id", "sex__tmp": "sex_id", "age__tmp": "age_group_id"}
    columns = []
    for column in model_settings["cascade_levels"].split(","):
        if column in rename:
            columns.append(rename[column])
        else:
            columns.append(column)
    df_input[columns].drop_duplicates().to_csv(
        model_settings["working_dir"] + "cascade_hierarchy.csv", index=False
    )


def get_combined_id(row: pd.Series) -> str:
    """Get combined age and location ID for age cascade."""
    return f"{row['age_group_id']}_{row['location_id']}"


def swimr_model(experiment_dir: Union[Path, str], submodel_id: str) -> None:
    """Run swimr model by submodel ID."""
    experiment_dir = Path(experiment_dir)
    swimr_dir = experiment_dir / "results" / "swimr"
    settings = load_settings(experiment_dir / "config" / "settings.yml")

    # Get submodel settings
    model_id = submodel_id.split("_")[0]
    param_id = int(submodel_id.split("_")[1][5:])
    subset_id = int(submodel_id.split("_")[2][6:])
    holdout_id = submodel_id.split("_")[3]
    model_settings = settings["swimr"]["models"][model_id]
    params = SwimrParams(model_id, param_sets=pd.read_csv(swimr_dir / "parameters.csv"))
    subsets = Subsets(
        model_id, model_settings, subsets=pd.read_csv(swimr_dir / "subsets.csv")
    )

    # Load data and filter by subset
    df_input = subsets.filter_subset(
        get_smoother_input("swimr", settings, experiment_dir, from_rover=True),
        subset_id,
    )
    breakpoint()
    df_input["holdout"] = df_input[settings["col_test"]] != 0
    if (
        model_settings["model_type"] == "cascade"
        and model_settings["cascade_levels"] == "age__tmp,locid"
    ):
        df_input["location_id"] = df_input.apply(get_combined_id, axis=1)
    if holdout_id != "full":
        df_input["holdout"] = df_input["holdout"] & (df_input[holdout_id] != 0)
    model_data = str(
        swimr_dir / "data" / f"{model_id}_subset{subset_id}_{holdout_id}.parquet"
    )

    # The swimr task script overwrites holdout with the value of holdout1, for some reason
    # So enforce the same column here
    if "holdout1" in df_input.columns:
        df_input["holdout1"] = df_input["holdout"].astype({"holdout": int})
    df_input.astype({"holdout": int}).to_parquet(model_data)

    # Create submodel settings
    model_settings["prediction_submodel_ids"] = get_str(
        df_input["location_id"].unique()
    )
    model_settings["prediction_age_group_ids"] = get_str(
        df_input["age_group_id"].unique()
    )
    model_settings["prediction_year_ids"] = get_str(df_input["year_id"].unique())
    model_settings["working_dir"] = str(swimr_dir / "submodels" / submodel_id) + "/"
    Path(model_settings["working_dir"]).mkdir(exist_ok=True)
    for param in params.params:
        if param in ("theta", "intercept_theta"):
            model_settings[param] = get_str(params.get_param(param, param_id))
        else:
            model_settings[param] = params.get_param(param, param_id)
    if (
        model_settings["model_type"] == "cascade"
        and "cascade_hierarchy_csv_path" not in model_settings
    ):
        create_cascade_hierarchy(df_input, model_settings)
        model_settings["cascade_hierarchy_csv_path"] = (
            model_settings["working_dir"] + "cascade_hierarchy.csv"
        )
    breakpoint()
    with open(model_settings["working_dir"] + "settings.yml", "w") as f:
        yaml.dump(model_settings, f)

    # Run submodel
    subprocess.run(
        [
            "/ihme/singularity-images/rstudio/shells/execRscript.sh",
            "-i",
            settings["swimr"]["singularity_image"],
            "-s",
            "/mnt/team/msca/priv/code/swimr/tasks/run_swimr.R",
            "--path_to_input_data",
            model_data,
            "--path_to_similarity_matrix",
            model_settings["similarity_matrix"],
            "--path_to_model_specs",
            model_settings["working_dir"] + "settings.yml",
            "--path_to_age_metadata",
            settings["swimr"]["age_metadata"],
            "--path_to_swimr_parentdir",
            settings["swimr"]["swimr_parentdir"],
            "--path_to_conda_binary",
            settings["swimr"]["conda_binary"],
            "--conda_env_name",
            settings["swimr"]["conda_env"],
        ],
        check=True,
    )

    # Get predictions
    df_pred = pd.read_parquet(model_settings["working_dir"] + "predictions.parquet")
    df_pred = df_pred.melt(
        id_vars=["age_group_id", "year_id"],
        value_vars=tuple(df_pred.columns[3:]),
        var_name="location_id",
        value_name="residual",
    )
    df_pred["model_id"] = model_id
    df_pred["param_id"] = param_id
    df_pred["holdout_id"] = holdout_id
    df_pred["location_id"] = pd.to_numeric(df_pred["location_id"])
    df_pred["sex_id"] = df_input["sex_id"].unique()[0]
    df_pred = df_pred.merge(
        right=df_input[as_list(settings["col_id"]) + ["test", settings["col_pred"]]],
        on=settings["col_id"],
    )
    df_pred[settings["col_pred"]] = df_pred.apply(
        lambda row: get_prediction(
            row, settings["col_pred"], settings["rover"]["model_type"]
        ),
        axis=1,
    )
    df_pred[
        as_list(settings["col_id"])
        + ["residual", settings["col_pred"], "model_id", "param_id", "holdout_id"]
    ].to_parquet(model_settings["working_dir"] + "predictions.parquet")


def main() -> None:
    fire.Fire(swimr_model)
