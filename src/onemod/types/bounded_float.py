from math import isnan, isinf
from typing_extensions import Annotated

from pydantic import BaseModel, Field, field_validator

class BoundedFloat(BaseModel):
    value: Annotated[float, Field]
    
    @classmethod
    def with_bounds(cls, ge: float | None = None, le: float | None = None, allow_nan: bool = False) -> BaseModel:
        """Factory method to create a custom bound BoundedFloat model"""
        field_args = {}
        if ge is None:
            field_args['le'] = le
        if le is None:
            field_args['ge'] = ge
            
        cls._allow_nan = allow_nan
        
        return Annotated[float, Field(strict=True, **field_args)]
    
    @field_validator('value')
    def check_nan(cls, value: float) -> float:
        if isinf(value):
            raise ValueError("Infinity values (inf, -inf) are not allowed.")
        
        if not cls._allow_nan and isnan(value):
            raise ValueError("NaN values are not allowed.")
        
        return value
