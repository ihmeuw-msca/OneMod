import pytest
from onemod.types.float import Float

def test_float_bound_within_bounds():
    StriclyPositiveFloat = Float.with_bounds(ge=0)
    positive_float = StriclyPositiveFloat(3.14)
    assert positive_float(3.14) == 3.14

def test_float_no_bounds():
    TestFloat = Float()
    test_float = TestFloat(3.141592653589793)
    assert test_float(3.141592653589793) == 3.141592653589793
    
def test_float_invalid_outside_bounds():
    TestFloat = Float.with_bounds(ge=-1.0, le=1.0)
    with pytest.raises(ValueError):
        out_of_bounds_float = TestFloat(1.01) # Greater than maximum

def test_float_invalid_nan():
    TestFloat = Float()
    with pytest.raises(ValueError):
        nan_float = TestFloat(float('nan')) # Not a number
        
def test_float_valid_nan():
    TestFloat = Float()
    nan_float = TestFloat(float('nan'), allow_nan=True)
    assert nan_float(float('nan')) == float('nan')
    
def test_float_invalid_inf():
    TestFloat = Float()
    with pytest.raises(ValueError):
        inf_float = TestFloat(float('inf')) # Infinity value
