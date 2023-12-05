from functools import partial
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from jobmon.client.task import Task

from onemod.actions.action import Action
from onemod.actions.rover_covsel_model import rover_covsel_model
from onemod.data.collect_results import collect_rover_covsel_results
from onemod.utils import get_rover_covsel_submodels

class RoverCovselApplication:
    """A RoverCovsel Application comprised of rover actions."""

    def __init__(self, experiment_dir: str | Path):
        """Create a RoverCovsel Application."""
        self.experiment_dir = experiment_dir
        self.submodels = get_rover_covsel_submodels(experiment_dir)

    def action_generator(self) -> Generator[Action, None, None]:
        """A generator to return actions to be run, with the correct dependencies.

        If run_local is True, returns a generator of partial functions.
        If false, return a generator of Jobmon tasks.
        """
        actions = []  # List of tasks to be returned
        for submodel_id in self.submodels:
            action = Action(
                rover_covsel_model,
                experiment_dir=self.experiment_dir,
                submodel_id=submodel_id
            )
            actions.append(action)
        actions.append(collect_rover_covsel_results(self.experiment_dir))

    def run(self) -> None:
        """Run the application in local mode."""
        for callable_action in self.action_generator():
            callable_action.evaluate()

    def build_task_dag(self) -> list["Task"]:
        """Build a list of tasks with upstreams, to pass to the scheduler."""
        task_list = [action.task for action in self.action_generator()]
        return task_list
