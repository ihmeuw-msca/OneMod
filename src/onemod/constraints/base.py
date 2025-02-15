from typing import Any, Callable

from polars import Series
from pydantic import BaseModel, Field, field_validator

from onemod.constraints.functions import bounds, is_in

# Global registry for constraints
CONSTRAINT_REGISTRY: dict[str, Callable] = {}


class Constraint(BaseModel):
    name: str
    args: dict[str, Any]

    func: Callable[[Series], None] | None = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        """Reconstruct the `func` attribute after deserialization."""
        self.func = CONSTRAINT_REGISTRY[self.name](**self.args)

    @field_validator("name")
    def validate_name(cls, value):
        """Ensure the constraint name is in the global registry."""
        if value not in CONSTRAINT_REGISTRY:
            raise ValueError(
                f"Unknown constraint: {value}. Did you forget to register it?"
            )
        return value

    def use_validation(self, column: Series) -> None:
        """Applies the constraint's validation function to a Polars Series."""
        if self.func is None:
            self.func = CONSTRAINT_REGISTRY[self.name](**self.args)
        self.func(column)

    def to_dict(self) -> dict:
        """Convert the constraint to a dictionary for serialization."""
        return {"name": self.name, "args": self.args}

    @classmethod
    def from_dict(cls, data: dict) -> "Constraint":
        """Reconstruct a Constraint from a dictionary."""
        name = data["name"]
        args = data["args"]
        return cls(name=name, args=args)


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
    ...     def use_validation(column: Series) -> None:
    ...         if not column.lt(limit).all():
    ...             raise ValueError(f"Values must be less than {limit}.")
    ...
    ...     return use_validation
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
