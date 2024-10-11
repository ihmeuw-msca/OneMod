import pytest

from onemod.constraints import Constraint

@pytest.mark.unit
def test_constraint_to_dict():
    constraint = Constraint("bounds", ge=0, le=1)
    assert constraint.to_dict() == {
        "name": "bounds",
        "args": {
            "ge": 0,
            "le": 1
        }
    }
