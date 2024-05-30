from pathlib import Path
from typing import Generator

from onemod.actions.action import Action
from onemod.actions.data.collect_results import collect_results
from onemod.actions.models.kreg_model import kreg_model
from onemod.application.base import Application
from onemod.utils import get_submodels


class KregApplication(Application):
    """A kreg application to run the kernel regression stage."""

    def __init__(self, directory: str | Path, max_attempts: int) -> None:
        """Create a kreg application."""
        self.directory = directory
        self.submodels = get_submodels("kreg", directory)
        self.max_attempts = max_attempts

    def action_generator(self) -> Generator[Action, None, None]:
        """A generator that returns actions to be run."""
        # Modeling tasks
        for submodel_id in self.submodels:
            action = Action(
                kreg_model,
                directory=self.directory,
                submodel_id=submodel_id,
                max_attempts=self.max_attempts,
            )
            yield action

        # Collection task
        yield Action(
            collect_results, stage_name="kreg", directory=self.directory
        )
