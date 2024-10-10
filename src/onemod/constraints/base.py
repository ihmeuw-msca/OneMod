from typing import Any, Callable, Dict
from polars import Series

from pydantic import BaseModel, field_validator

from .functions import bounds, is_in


class Constraint(BaseModel):
    name: str
    args: Dict[str, Any]
    
    func: Callable[[Series], None] = None
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, args=kwargs)
        self.func = CONSTRAINT_REGISTRY[name](**kwargs)

    @field_validator('name')
    def validate_name(cls, value):
        """Ensure the constraint name is in the global registry."""
        if value not in CONSTRAINT_REGISTRY:
            raise ValueError(f"Unknown constraint: {value}. Did you forget to register it?")
        return value

    def validate(self, column: Series) -> None:
        """Applies the constraint's validation function to a Polars Series."""
        if self.func is None:
            self.func = CONSTRAINT_REGISTRY[self.name](**self.args)
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
        return cls(name=name, args=args)
    
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

def register_preset_constraints() -> None:
    """Registers preset constraint functions."""
    register_constraint("bounds", bounds)
    register_constraint("is_in", is_in)
    
register_preset_constraints()
