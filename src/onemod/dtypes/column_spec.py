from typing import Type

from pydantic import BaseModel, field_serializer, field_validator

from onemod.constraints.base import Constraint


class ColumnSpec(BaseModel):
    type: (
        Type[int]
        | Type[float]
        | Type[str]
        | Type[bool]
        | Type[BaseModel]
        | None
    ) = None
    constraints: list[Constraint] | None = None

    @field_serializer("type")
    def serialize_type(self, t, info):
        return t.__name__ if t else None

    @field_validator("type", mode="before")
    def deserialize_type(cls, v):
        if isinstance(v, str):
            type_mapping = {
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
            }
            if v in type_mapping:
                return type_mapping[v]
            else:
                raise ValueError(f"Unknown type string: {v}")
        return v

    @field_serializer("constraints")
    def serialize_constraints(self, constraints):
        if constraints is None:
            return None
        return [constraint.model_dump() for constraint in constraints]
