from typing import Callable, Dict, Any
from polars import Series


class Constraint:
    def __init__(self, name: str, func: Callable[[Series], None], args: Dict[str, Any]):
        """
        Initialize a Constraint object.

        Parameters
        ----------
        name : str
            The name of the constraint (e.g., "bounds", "not_empty").
        func : callable
            The validation function to apply.
        args : dict
            The arguments used to define the constraint (e.g., ge, le).
        """
        self.name = name
        self.func = func
        self.args = args

    def validate(self, column: Series) -> None:
        """Applies the constraint's validation function to a Polars Series."""
        self.func(column)

    def to_dict(self) -> dict:
        """Convert the constraint to a dictionary for serialization."""
        return {
            "name": self.name,
            "args": self.args
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Constraint':
        """Reconstruct a Constraint from a dictionary."""
        name = data["name"]
        args = data["args"]
        if name not in CONSTRAINT_REGISTRY:
            raise ValueError(f"Unknown constraint: {name}. Did you forget to register it?")
        func = CONSTRAINT_REGISTRY[name](**args)
        return cls(name=name, func=func, args=args)
    
# Global registry for constraints
CONSTRAINT_REGISTRY: Dict[str, Callable] = {}

def register_constraint(name: str, func: Callable) -> None:
    """
    Allows users to register custom constraint functions.
    
    Parameters
    ----------
    name : str
        The name of the constraint.
    func : callable
        The validation function to apply.
        
    Examples
    --------
    >>> def custom_constraint_example(limit: int) -> Callable:
    ...     def validate(column: Series) -> None:
    ...         if not column.lt(limit).all():
    ...             raise ValueError(f"Values must be less than {limit}.")
    ...     return validate
    >>> register_constraint("custom_constraint", custom_constraint_example)
    """
    if name in CONSTRAINT_REGISTRY:
        raise ValueError(f"Constraint '{name}' is already registered.")
    CONSTRAINT_REGISTRY[name] = func
