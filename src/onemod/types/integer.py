from typing_extensions import Annotated

from pydantic import BaseModel, Field


class Integer(BaseModel):
    value: Annotated[int, Field]
    
    @classmethod
    def with_bounds(cls, ge: int | None = None, le: int | None = None) -> BaseModel:
        """Factory method to create a custom bounded Integer model"""
        field_args = {}
        if ge is None:
            field_args['le'] = le
        if le is None:
            field_args['ge'] = ge
        
        return Annotated[int, Field(strict=True, **field_args)]
