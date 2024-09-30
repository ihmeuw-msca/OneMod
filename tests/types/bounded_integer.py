import pytest
from onemod.types.bounded_integer import BoundedInteger

def test_integer_bound_within_bounds():
    StriclyPositiveInteger = BoundedInteger.with_bounds(ge=0)
    bounded_integer = StriclyPositiveInteger(7)
    assert bounded_integer(7) == 7

def test_integer_bound_outside_bounds():
    TestInteger = BoundedInteger.with_bounds(ge=-100, le=100)
    with pytest.raises(ValueError):
        out_of_bounds_integer = TestInteger(-101) # Less than minimum
        
def test_integer_non_integer():
    TestInteger = BoundedInteger.with_bounds(ge=0, le=10)
    with pytest.raises(ValueError):
        non_integer = TestInteger(3.14) # Not an integer

def test_integer_no_bounds():
    TestInteger = BoundedInteger()
    bounded_integer = TestInteger(42)
    assert bounded_integer(42) == 42
