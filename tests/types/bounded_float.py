import pytest
from onemod.types.bounded_float import BoundedFloat

def test_float_bound_within_bounds():
    StriclyPositiveFloat = BoundedFloat.with_bounds(ge=0)
    bounded_float = StriclyPositiveFloat(3.14)
    assert bounded_float(3.14) == 3.14

def test_float_no_bounds():
    TestFloat = BoundedFloat()
    bounded_float = TestFloat(3.141592653589793)
    assert bounded_float(3.141592653589793) == 3.141592653589793
    
def test_float_invalid_outside_bounds():
    TestFloat = BoundedFloat.with_bounds(ge=-1.0, le=1.0)
    with pytest.raises(ValueError):
        out_of_bounds_float = TestFloat(1.01) # Greater than maximum

def test_float_invalid_nan():
    TestFloat = BoundedFloat()
    with pytest.raises(ValueError):
        nan_float = TestFloat(float('nan')) # Not a number
        
def test_float_valid_nan():
    TestFloat = BoundedFloat()
    nan_float = TestFloat(float('nan'), allow_nan=True)
    assert nan_float(float('nan')) == float('nan')
    
def test_float_invalid_inf():
    TestFloat = BoundedFloat()
    with pytest.raises(ValueError):
        inf_float = TestFloat(float('inf')) # Infinity value
