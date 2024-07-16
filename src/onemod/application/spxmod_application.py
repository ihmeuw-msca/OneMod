from pathlib import Path
from typing import Generator

from onemod.actions.action import Action
from onemod.actions.data.collect_results import collect_results
from onemod.actions.models.spxmod_model import spxmod_model
from onemod.application.base import Application
from onemod.utils import get_submodels


class SPxModApplication(Application):
    """An SPxMod application to run the spxmod stage."""

    def __init__(self, directory: str | Path, max_attempts: int) -> None:
        """Create an SPxMod Application."""
        self.directory = directory
        self.submodels = get_submodels("spxmod", directory)
        self.max_attempts = max_attempts

    def action_generator(self) -> Generator[Action, None, None]:
        """A generator to return actions to be run."""
        # Modeling tasks
        for submodel_id in self.submodels:
            action = Action(
                spxmod_model,
                directory=self.directory,
                submodel_id=submodel_id
            )
            yield action

        # Collection task
        yield Action(
            collect_results, stage_name="spxmod", directory=self.directory
        )
