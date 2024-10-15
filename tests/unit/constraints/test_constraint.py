import pytest

from onemod.constraints import Constraint

@pytest.mark.unit
def test_constraint_model():
    constraint = Constraint(name="bounds", args=dict(ge=0, le=1))
    
    expected = {
        "name": "bounds",
        "args": {
            "ge": 0,
            "le": 1
        }
    }
    
    actual = constraint.model_dump()
    
    assert actual == expected
