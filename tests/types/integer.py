import pytest
from onemod.types.integer import Integer

def test_integer_bound_within_bounds():
    StriclyPositiveInteger = Integer.with_bounds(ge=0)
    positive_integer = StriclyPositiveInteger(7)
    assert positive_integer(7) == 7

def test_integer_bound_outside_bounds():
    TestInteger = Integer.with_bounds(ge=-100, le=100)
    with pytest.raises(ValueError):
        out_of_bounds_integer = TestInteger(-101)  # Less than minimum
        
def test_integer_non_integer():
    TestInteger = Integer.with_bounds(ge=0, le=10)
    with pytest.raises(ValueError):
        non_integer = TestInteger(3.14)  # Not an integer

def test_integer_no_bounds():
    TestInteger = Integer()
    test_integer = TestInteger(42)
    assert test_integer(42) == 42
