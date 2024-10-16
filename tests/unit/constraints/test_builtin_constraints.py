from polars import Series
import pytest

from onemod.constraints import bounds, no_inf

@pytest.mark.unit
def test_within_bounds():
    test_series = Series([0, 1, 2, 3, 4, 5, 100])
    validate = bounds(ge=0, le=100)
    validate(test_series)  # No error
    
@pytest.mark.unit
def test_invalid_outside_bounds():
    test_series = Series([-1.1, -0.5, 0, 0.5, 1.0])
    validate = bounds(ge=-1, le=1)
    with pytest.raises(ValueError):
        validate(test_series)  # ValueError: All values must be greater than or equal to 0.

@pytest.mark.unit
def test_no_inf_values():
    test_series = Series([1, 2, 3, 4, 5])
    validate = no_inf()
    validate(test_series)
    
@pytest.mark.unit
def test_no_inf_with_inf():
    test_series = Series([1.0, 2.0, float('inf'), 4.0, 5.0])
    validate = no_inf()
    with pytest.raises(ValueError):
        validate(test_series)  # ValueError: All values must be finite.
        
@pytest.mark.unit
def test_no_inf_with_neg_inf():
    test_series = Series([1.0, 2.0, float('-inf'), 4.0, 5.0])
    validate = no_inf()
    with pytest.raises(ValueError):
        validate(test_series)  # ValueError: All values must be finite.
