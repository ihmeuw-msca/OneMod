from pathlib import Path
from typing import Generator

from onemod.actions.action import Action
from onemod.actions.data.collect_results import collect_results
from onemod.actions.models.rover_covsel_model import rover_covsel_model
from onemod.application.application import Application
from onemod.utils import get_submodels


class RoverCovselApplication(Application):
    """A RoverCovsel Application comprised of rover actions."""

    def __init__(self, directory: str | Path, max_attempts: int) -> None:
        """Create a RoverCovsel Application."""
        self.directory = directory
        self.submodels = get_submodels("rover_covsel", directory)
        self.max_attempts = max_attempts

    def action_generator(self) -> Generator[Action, None, None]:
        """A generator to return actions to be run."""
        # Modeling tasks
        for submodel_id in self.submodels:
            action = Action(
                rover_covsel_model,
                directory=self.directory,
                submodel_id=submodel_id,
                max_attempts=self.max_attempts,
            )
            yield action

        # Collection task
        yield Action(
            collect_results,
            stage_name="rover_covsel",
            directory=self.directory,
        )
