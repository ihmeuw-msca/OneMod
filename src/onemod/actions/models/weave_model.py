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
    get_weave_input,
    parse_weave_submodel,
)


def weave_model(directory: str, submodel_id: str) -> None:
    """Run weave model by submodel ID.

    Parameters
    ----------
    directory
        The path to the directory containing the experiment data.
    submodel_id (str)
        The ID of the submodel to be processed.

    """
    dataif, config = get_handle(directory)

    # Get submodel settings
    submodel = parse_weave_submodel(submodel_id)
    model_config = config.weave.models[submodel["model_id"]]
    params = WeaveParams(
        submodel["model_id"], param_sets=dataif.load_weave("parameters.csv")
    )
    subsets = WeaveSubsets(
        submodel["model_id"],
        model_config,
        subsets=dataif.load_weave("subsets.csv"),
    )

    # Load data and filter by subset and batch
    df_input = subsets.filter_subset(
        get_weave_input(config, dataif),
        submodel["subset_id"],
        submodel["batch_id"],
    ).rename(columns={"batch": "predict"})
    df_input["fit"] = df_input[config.test] == 0
    if submodel["holdout_id"] != "full":
        df_input["fit"] = df_input["fit"] & (
            df_input[submodel["holdout_id"]] == 0
        )
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
                    params.get_param(
                        f"{dim_name}__distance_dict", submodel["param_id"]
                    ),
                    allow_pickle=True,
                )
            else:
                dim_dict[param] = params.get_param(
                    f"{dim_name}__{param}", submodel["param_id"]
                )
        dimensions.append(Dimension(**dim_dict))
    smoother = Smoother(dimensions)

    # WeAve models throw error if data contains NaNs
    # Replace possible NaNs with dummy value
    for column in ["spxmod_value", "spxmod_se"]:
        df_input.loc[
            df_input.eval(f"fit == False and {column}.isna()"), column
        ] = 1

    # Get predictions
    logger.info(f"Fitting smoother for {submodel_id=}")
    df_pred = smoother(
        data=df_input,
        observed="spxmod_value",
        stdev="spxmod_se",
        smoothed="residual",
        fit="fit",
        predict="predict",
    ).rename(columns={"residual_sd": "residual_se"})
    logger.info(f"Completed fitting, predicting for {submodel_id=}")
    df_pred[config.pred] = df_pred.apply(
        lambda row: get_prediction(row, config.pred, config.mtype),
        axis=1,
    )
    df_pred = df_pred[config.ids + ["residual", "residual_se", config.pred]]
    dataif.dump_weave(df_pred, f"submodels/{submodel_id}.parquet")


def main() -> None:
    """Main entry point of the module.

    This function uses the Fire library to allow the weave_model
    function to be invoked from the command line.

    """
    fire.Fire(weave_model)
