import pytest

from onemod.constraints.base import CONSTRAINT_REGISTRY, register_constraint

@pytest.mark.unit
def test_register_constraint():
    def constraint_func(x):
        """Returns a function which checks whether all items are equal to x."""
        def validate(series):
            if not (series == x).all():
                raise ValueError(f"Expected all items to be {x}")
        return validate
    
    register_constraint("equal_to", constraint_func)
    # Check that the function is registered
    assert "equal_to" in CONSTRAINT_REGISTRY
        