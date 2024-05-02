"""Run weave model."""

import fire
import numpy as np
from loguru import logger
from weave.dimension import Dimension
from weave.smoother import Smoother

from onemod.utils import (
    WeaveParams,
    WeaveSubsets,
    get_handle,
    get_prediction,
    get_smoother_input,
)


def weave_model(directory: str, submodel_id: str) -> None:
    """Run weave model by submodel ID.

    Parameters
    ----------
    directory
        The path to the directory containing the experiment data.
    submodel_id (str)
        The ID of the submodel to be processed.

    Notes
    -----
    Each submodel_id has the form:

    f"{model_id}__param{param_id}__subset{subset_id}__{holdout_id}__batch{batch_id}"

    with:
    * model_id (str): user input
    * param_id (int): assigned by WeaveParams
    * subset_id (int): assigned by WeaveSubsets
    * holdout_id (str): user input
    * batch_id (int): assigned by WeaveSubsets

    Example: "age_model__param0__subset0__holdout1__batch0"

    """
    dataif, config = get_handle(directory)

    # Get submodel settings
    model_id = submodel_id.split("__")[0]
    param_id = int(submodel_id.split("__")[1][5:])
    subset_id = int(submodel_id.split("__")[2][6:])
    holdout_id = submodel_id.split("__")[3]
    batch_id = int(submodel_id.split("__")[4][5:])
    model_config = config.weave.models[model_id]
    params = WeaveParams(
        model_id, param_sets=dataif.load_weave("parameters.csv")
    )
    subsets = WeaveSubsets(
        model_id, model_config, subsets=dataif.load_weave("subsets.csv")
    )

    # Load data and filter by subset and batch
    df_input = subsets.filter_subset(
        get_smoother_input("weave", config, dataif),
        subset_id,
        batch_id,
    ).rename(columns={"batch": "predict"})
    df_input["fit"] = df_input[config.test] == 0
    if holdout_id != "full":
        df_input["fit"] = df_input["fit"] & (df_input[holdout_id] == 0)
    df_input = df_input[df_input["fit"] | df_input["predict"]].drop(
        columns=[config.test] + config.holdouts
    )

    # Create smoother objects
    dimensions = []
    for dim_name, dim_object in model_config.dimensions.items():
        dim_dict = dim_object.model_dump()
        for param in params.get_dimension_params(dim_object):
            if param == "distance_dict":
                dim_dict["distance_dict"] = np.load(
                    params.get_param(f"{dim_name}__distance_dict", param_id),
                    allow_pickle=True,
                )
            else:
                dim_dict[param] = params.get_param(
                    f"{dim_name}__{param}", param_id
                )
        dimensions.append(Dimension(**dim_dict))
    smoother = Smoother(dimensions)

    # WeAve models throw error if data contains NaNs
    # Replace possible NaNs with dummy value
    for column in ["residual_value", "residual_se"]:
        df_input.loc[
            df_input.eval(f"fit == False and {column}.isna()"), column
        ] = 1

    # Get predictions
    logger.info(f"Fitting smoother for {submodel_id=}")
    df_pred = smoother(
        data=df_input,
        observed="residual_value",
        stdev="residual_se",
        smoothed="residual",
        fit="fit",
        predict="predict",
    )
    logger.info(f"Completed fitting, predicting for {submodel_id=}")
    df_pred[config.pred] = df_pred.apply(
        lambda row: get_prediction(row, config.pred, config.mtype),
        axis=1,
    )
    df_pred["model_id"] = model_id
    df_pred["param_id"] = param_id
    df_pred["holdout_id"] = holdout_id
    df_pred = df_pred[
        config.ids
        + ["residual", config.pred, "model_id", "param_id", "holdout_id"]
    ]
    dataif.dump_weave(df_pred, f"submodels/{submodel_id}.parquet")


def main() -> None:
    """Main entry point of the module.

    This function uses the Fire library to allow the weave_model
    function to be invoked from the command line.

    """
    fire.Fire(weave_model)
