"""Run rover covariate selection model."""
import fire
from pathlib import Path
from modrover.api import Rover
from pplkit.data.interface import DataInterface
from onemod.utils import get_rover_covsel_input, Subsets


def rover_covsel_model(experiment_dir: Path | str, submodel_id: str) -> None:
    """Run rover covariate selection model by submodel ID.

    Parameters
    ----------
    experiment_dir
        Parent folder where the experiment is run.
        - ``experiment_dir / config / settings.yaml`` contains rover modeling settings
        - ``experiment_dir / results / rover`` stores all rover results
    submodel_id
        Example of ``submodel_id`` can be written as ``'subset0'``. In this case
        the numbered id ``0`` will be used to lookup the corresponding subsets
        stored in ``subsets.csv``.

    Outputs
    -------
    rover.pkl
        Rover object used for plotting and diagnostics.
    learner_info.csv
        Information about every learner.
    summary.csv
        Summary covariate coefficients from the ensemble model.

    """
    dataif = DataInterface(experiment=experiment_dir)
    dataif.add_dir("config", dataif.experiment / "config")
    dataif.add_dir("covsel", dataif.experiment / "results" / "rover_covsel")
    settings = dataif.load_config("settings.yml")

    subsets = Subsets(
        "rover_covsel",
        settings["rover_covsel"],
        subsets=dataif.load_covsel("subsets.csv"),
    )

    # Load and filter by subset
    subset_id = int(submodel_id[6:])
    df_input = subsets.filter_subset(get_rover_covsel_input(settings), subset_id)

    # Create a test column if not existing
    test_col = settings["col_test"]
    if test_col not in df_input:
        df_input[test_col] = df_input[settings["col_obs"]].isna().astype("int")

    df_train = df_input[df_input[settings["col_test"]] == 0]

    dataif.dump_covsel(df_train, f"data/{submodel_id}.parquet")

    # Create rover objects
    rover = Rover(**settings["rover_covsel"]["Rover"])

    # Fit rover model
    rover.fit(data=df_train, **settings["rover_covsel"]["Rover.fit"])

    # Save results
    dataif.dump_covsel(rover, f"submodels/{submodel_id}/rover.pkl")
    dataif.dump_covsel(rover.learner_info, f"submodels/{submodel_id}/learner_info.csv")
    dataif.dump_covsel(rover.summary, f"submodels/{submodel_id}/summary.csv")


def main() -> None:
    fire.Fire(rover_covsel_model)
