from pathlib import Path
from typing import Generator

from onemod.actions.action import Action
from onemod.actions.data.collect_results import collect_results
from onemod.actions.models.rover_covsel_model import rover_covsel_model
from onemod.application.base import Application
from onemod.utils import get_rover_covsel_submodels


class RoverCovselApplication(Application):
    """A RoverCovsel Application comprised of rover actions."""

    def __init__(self, experiment_dir: str | Path):
        """Create a RoverCovsel Application."""
        self.experiment_dir = experiment_dir
        self.submodels = get_rover_covsel_submodels(experiment_dir)

    def action_generator(self) -> Generator[Action, None, None]:
        """A generator to return actions to be run."""
        for submodel_id in self.submodels:
            action = Action(
                rover_covsel_model,
                experiment_dir=self.experiment_dir,
                submodel_id=submodel_id,
            )
            yield action
        yield Action(
            collect_results,
            stage_name="rover_covsel",
            experiment_dir=self.experiment_dir,
        )
