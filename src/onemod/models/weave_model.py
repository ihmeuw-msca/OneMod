"""Run weave model."""
import fire
import numpy as np
from weave.dimension import Dimension
from weave.smoother import Smoother

from onemod.utils import (
    as_list,
    get_handle,
    get_prediction,
    get_smoother_input,
    Subsets,
    WeaveParams,
)

# weave kernel parameters
kernel_params = {
    "exponential": "radius",
    "tricubic": "exponent",
    "depth": "radius",
}


def weave_model(experiment_dir: str, submodel_id: str) -> None:
    """Run weave model by submodel ID.

    Args:
        experiment_dir (Union[Path, str]): The path to the directory containing the
            experiment data.
        submodel_id (str): The ID of the submodel to be processed.
    """
    dataif, settings = get_handle(experiment_dir)

    # Get submodel settings
    model_id = submodel_id.split("_")[0]
    param_id = int(submodel_id.split("_")[1][5:])
    subset_id = int(submodel_id.split("_")[2][6:])
    holdout_id = submodel_id.split("_")[3]
    batch_id = int(submodel_id.split("_")[4][5:])
    model_settings = settings["weave"]["models"][model_id]
    params = WeaveParams(model_id, param_sets=dataif.load_weave("parameters.csv"))
    subsets = Subsets(
        model_id, model_settings, subsets=dataif.load_weave("subsets.csv")
    )

    # Load data and filter by subset and batch
    df_input = subsets.filter_subset(
        get_smoother_input("weave", config=settings, dataif=dataif, from_rover=True),
        subset_id,
        batch_id,
    ).rename(columns={"batch": "predict"})
    df_input["fit"] = df_input[settings["col_test"]] == 0
    if holdout_id != "full":
        df_input["fit"] = df_input["fit"] & (df_input[holdout_id] == 0)
    df_input = df_input[df_input["fit"] | df_input["predict"]].drop(
        columns=as_list(settings["col_test"]) + as_list(settings["col_holdout"])
    )

    # Create smoother objects
    dimensions = []
    for dim_name, dim_dict in model_settings["dimensions"].items():
        dim_dict = dim_dict.model_dump(
            exclude={"parent_args"}
        )  # Base type is a pydantic model, so convert to dict
        if dim_dict["kernel"] != "identity":
            param = kernel_params[dim_dict["kernel"]]
            dim_dict[param] = params.get_param(f"{dim_name}_{param}", param_id)
        if "distance" in dim_dict:
            if dim_dict["distance"] == "dictionary":
                dim_dict["distance_dict"] = np.load(
                    params.get_param(f"{dim_name}_distance_dict", param_id),
                    allow_pickle=True,
                ).item()
        dimensions.append(Dimension(**dim_dict))
    smoother = Smoother(dimensions)

    # Impute missing observation and holdout values
    # Assume that missing numbers for holdouts are implicit 1's (i.e. used for testing anyways)
    df_input.fillna(1, inplace=True)
    # Get predictions
    df_pred = smoother(
        data=df_input,
        observed="residual_value",
        stdev="residual_se",
        smoothed="residual",
        fit="fit",
        predict="predict",
    )
    df_pred[settings["col_pred"]] = df_pred.apply(
        lambda row: get_prediction(row, settings["col_pred"], settings["model_type"]),
        axis=1,
    )
    df_pred["model_id"] = model_id
    df_pred["param_id"] = param_id
    df_pred["holdout_id"] = holdout_id
    df_pred = df_pred[
        as_list(settings["col_id"])
        + ["residual", settings["col_pred"], "model_id", "param_id", "holdout_id"]
    ]
    dataif.dump_weave(df_pred, f"submodels/{submodel_id}.parquet")


def main() -> None:
    """Main entry point of the module.

    This function uses the Fire library to allow the weave_model function to be
    invoked from the command line.
    """
    fire.Fire(weave_model)
