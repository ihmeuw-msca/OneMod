"""Run rover covariate selection model."""

import fire
from loguru import logger
from modrover.api import Rover

from onemod.utils import Subsets, get_handle


def rover_covsel_model(directory: str, submodel_id: str) -> None:
    """Run rover covariate selection model by submodel ID.

    Parameters
    ----------
    directory
        Parent folder where the experiment is run.
        - ``directory / config / settings.yaml`` contains rover modeling settings
        - ``directory / results / rover_covsel`` stores all rover results
    submodel_id
        Example of ``submodel_id`` can be written as ``'subset0'``.
        In this case the numbered id ``0`` will be used to lookup the
        corresponding subsets stored in ``subsets.csv``.

    Outputs
    -------
    rover.pkl
        Rover object used for plotting and diagnostics.
    learner_info.csv
        Information about every learner.
    summary.csv
        Summary covariate coefficients from the ensemble model.

    """
    dataif, config = get_handle(directory)
    stage_config = config.rover_covsel

    subsets = Subsets(
        "rover_covsel",
        stage_config,
        subsets=dataif.load_rover_covsel("subsets.csv"),
    )

    # Load and filter by subset
    subset_id = int(submodel_id.removeprefix("subset"))
    df_input = subsets.filter_subset(dataif.load_data(), subset_id)
    df_train = df_input[df_input[config.test] == 0]

    # Fit model
    if len(df_train) > 0:
        logger.info(f"Fitting rover for {submodel_id=}")
        dataif.dump_rover_covsel(df_train, f"data/{submodel_id}.parquet")

        # Create rover objects
        rover_init = stage_config.rover
        rover = Rover(
            obs=config.obs,
            model_type=config.mtype,
            cov_fixed=rover_init.cov_fixed,
            cov_exploring=rover_init.cov_exploring,
            weights=config.weights,
            holdouts=config.holdouts,
        )

        # Fit rover model
        logger.info(
            f"Fitting the rover model with options {stage_config.rover_fit}"
        )
        rover.fit(data=df_train, **stage_config.rover_fit.model_dump())

        # Save results
        logger.info("Saving rover results after fitting")
        dataif.dump_rover_covsel(rover, f"submodels/{submodel_id}/rover.pkl")
        dataif.dump_rover_covsel(
            rover.learner_info, f"submodels/{submodel_id}/learner_info.csv"
        )
        dataif.dump_rover_covsel(
            rover.summary, f"submodels/{submodel_id}/summary.csv"
        )
    else:
        logger.info(f"No training data for {submodel_id=}, skipping model")


def main() -> None:
    fire.Fire(rover_covsel_model)
