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
        func_name = self.func.__name__
        # Might need to think of a better solution. But collect_results is a single action
        # that runs at multiple points in the DAG, with different stage names.

        # Need to append to the name for lookup.
        if func_name == 'collect_results':
            func_name += self.kwargs['stage_name']

        return func_name

    def evaluate(self):
        """Evaluate the action."""
        return self.func(*self.args, **self.kwargs)
