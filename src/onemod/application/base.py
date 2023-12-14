from abc import abstractmethod
from typing import Generator

from onemod.actions.action import Action


class Application:

    @abstractmethod
    def action_generator(self) -> Generator[Action, None, None]:
        pass

    def run(self) -> None:
        """Evaluate an application in memory."""
        for action in self.action_generator():
            action.evaluate()
