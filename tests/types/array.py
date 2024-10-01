import pytest
import numpy as np
from pydantic import ValidationError
from onemod.types.array import Array

def test_array_with_valid_input():
    schema = Array.of_type(np.int32, shape=(3,))

    valid_array = np.array([1, 2, 3], dtype=np.int32)
    
    schema(value=valid_array)  # No error should be raised

def test_array_with_type_mismatch():
    schema = Array.of_type(np.int32, shape=(3,))

    invalid_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Wrong type
    
    with pytest.raises(ValidationError) as excinfo:
        schema(value=invalid_array)
    assert "Expected array of type" in str(excinfo.value)

def test_array_with_shape_mismatch():
    schema = Array.of_type(np.int32, shape=(3,))

    invalid_array = np.array([[1, 2], [3, 4]], dtype=np.int32)  # Wrong shape
    
    with pytest.raises(ValidationError) as excinfo:
        schema(value=invalid_array)
    assert "Expected array shape" in str(excinfo.value)
