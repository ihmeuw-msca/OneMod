from typing import Any, Callable, List, Type, Union
from typing_extensions import TypedDict

from pydantic import BaseModel


class ColumnSpec(TypedDict, total=False):
    type: Union[Type[int], Type[float], Type[str], Type[bool], Type[BaseModel]]
    constraints: List[Callable[[Any], bool]]
