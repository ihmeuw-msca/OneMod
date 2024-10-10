from typing import List, Type, Union
from typing_extensions import TypedDict

from pydantic import BaseModel

from onemod.constraints.base import Constraint


class ColumnSpec(TypedDict, total=False):
    type: Union[Type[int], Type[float], Type[str], Type[bool], Type[BaseModel]]
    constraints: List[Constraint]
