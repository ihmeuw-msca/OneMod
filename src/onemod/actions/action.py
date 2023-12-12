from typing import Callable


class Action:
    """Wrapper for actions.

    Allows local execution as well as instantiation of a Jobmon task.

    An action has 2 key use cases:
    1. Be run as a callable
    2. Be added to a Jobmon task dag
    """

    def __init__(self, func: Callable, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return self.func.__name__

    def evaluate(self):
        """Evaluate the action."""
        return self.func(*self.args, **self.kwargs)
