import numpy as np
from typing import Type

from pydantic import BaseModel, field_validator


class Array(BaseModel):
    value: np.ndarray

    @classmethod
    def of_type(cls, item_type: Type, shape: tuple[int, ...] = None) -> Type:
        """Creates a custom validator for a NumPy array with a specific type and shape."""
        array_type = cls

        @field_validator('value')
        def check_array_type(cls, v):
            if not isinstance(v, np.ndarray):
                raise TypeError("Expected a NumPy array")
            if shape and v.shape != shape:
                raise ValueError(f"Expected array shape {shape}, got {v.shape}")
            if not issubclass(v.dtype.type, item_type):
                raise TypeError(f"Expected array of type {item_type}, got {v.dtype.type}")
            return v

        return array_type
