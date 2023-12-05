from functools import partial
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from jobmon.client.task import Task

from onemod.action.rover_covsel_model import rover_covsel_model
from onemod.data.collect_results import collect_rover_covsel_results
from onemod.utils import get_rover_covsel_submodels

class RoverCovselApplication:
    """A RoverCovsel Application comprised of rover actions."""

    def __init__(self, experiment_dir: str | Path):
        """Create a RoverCovsel Application."""
        self.experiment_dir = experiment_dir
        self.submodels = get_rover_covsel_submodels(experiment_dir)
        self.tool

    def action_generator(
        self, run_local: bool = True
    ) -> Generator[partial | "Task", None, None]:
        """A generator to return actions to be run, with the correct dependencies.

        If run_local is True, returns a generator of partial functions.
        If false, return a generator of Jobmon tasks.
        """
        actions = []  # List of tasks to be returned
        for submodel_id in self.submodels:
            action = rover_covsel_model(submodel_id=submodel_id)
            actions.append(action)
        actions.append(collect_rover_covsel_results())

    def run(self) -> None:
        """Run the application in local mode."""
        for callable_action in self.action_generator():
            callable_action()

    def build_task_dag(self) -> list["Task"]:
        """Build a list of tasks with upstreams, to pass to the scheduler."""
        task_list = list(self.action_generator(run_local=False))
        return task_list

