from pathlib import Path
from typing import Generator

from onemod.actions.action import Action
from onemod.actions.models.regmod_smooth_model import regmod_smooth_model
from onemod.application.base import Application
from onemod.actions.data.collect_results import collect_results


class RegmodSmoothApplication(Application):
    """A RegmodSmooth Application to run through the regmod smooth stage."""

    def __init__(self, experiment_dir: str | Path):
        """Create a Regmod Smooth Application."""
        self.experiment_dir = experiment_dir

    def action_generator(self) -> Generator[Action, None, None]:
        """A generator to return actions to be run, with the correct dependencies.

        For the regmod smooth stage, there are currently just two actions - model and plot.
        """
        # Modeling task
        yield Action(
            regmod_smooth_model,
            experiment_dir=self.experiment_dir,
        )
        # Collection task
        yield Action(
            collect_results,
            stage_name="regmod_smooth",
            experiment_dir=self.experiment_dir
        )
