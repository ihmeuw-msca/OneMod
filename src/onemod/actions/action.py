from typing import Callable


class Action:
    """Wrapper for actions.

    Allows local execution as well as instantiation of a Jobmon task.

    An action has 2 key use cases:
    1. Be run as a callable
    2. Be added to a Jobmon task dag
    """

    def __init__(
        self, func: Callable, entrypoint: str | None = None, *args, **kwargs
    ) -> None:
        self.func = func
        self._entrypoint = entrypoint
        self.args = args
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return self.func.__name__

    @property
    def entrypoint(self) -> str:
        """For most actions we can simply use the function name. However, for collect results,
        since there is a single entrypoint for different stages, we need to specify.
        """
        if self._entrypoint:
            return self._entrypoint
        else:
            return self.name

    def evaluate(self) -> None:
        """Evaluate the action."""
        return self.func(*self.args, **self.kwargs)
