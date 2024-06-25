from pathlib import Path
from typing import Generator

from onemod.actions.action import Action
from onemod.actions.models.ensemble_model import ensemble_model
from onemod.application.base import Application


class EnsembleApplication(Application):
    """An application to run the ensemble stage."""

    def __init__(self, directory: str | Path, max_attempts: int) -> None:
        self.directory = directory
        self.max_attempts = max_attempts

    def action_generator(self) -> Generator[Action, None, None]:
        """A generator to return actions to be run."""
        yield Action(
            ensemble_model,
            directory=self.directory,
            max_attempts=self.max_attempts,
        )
