from pathlib import Path
from typing import Generator

from onemod.actions.action import Action
from onemod.actions.data.collect_results import collect_results
from onemod.actions.models.weave_model import weave_model
from onemod.application.base import Application
from onemod.utils import get_weave_submodels


class WeaveApplication(Application):
    """An application to run weave."""

    def __init__(self, experiment_dir: str | Path):
        self.experiment_dir = experiment_dir
        self.submodels = get_weave_submodels(experiment_dir)

    def action_generator(self) -> Generator[Action, None, None]:
        """A generator to return actions to be run."""
        for submodel_id in self.submodels:
            action = Action(
                weave_model, experiment_dir=self.experiment_dir, submodel_id=submodel_id
            )
            yield action
        yield Action(
            collect_results, stage_name="weave", experiment_dir=self.experiment_dir
        )
