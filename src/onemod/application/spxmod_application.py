from pathlib import Path
from typing import Generator

from onemod.actions.action import Action
from onemod.actions.data.collect_results import collect_results
from onemod.actions.models.spxmod_model import spxmod_model
from onemod.application.base import Application


class SPxModApplication(Application):
    """A SPxMod Application to run through the regmod smooth stage."""

    def __init__(self, directory: str | Path):
        """Create a SPxMod Application."""
        self.directory = directory

    def action_generator(self) -> Generator[Action, None, None]:
        """A generator to return actions to be run, with the correct dependencies.

        For the regmod smooth stage, there are currently just two actions - model and plot.
        """
        # Modeling task
        yield Action(
            spxmod_model,
            directory=self.directory,
        )
        # Collection task
        yield Action(
            collect_results,
            stage_name="spxmod",
            directory=self.directory,
        )
