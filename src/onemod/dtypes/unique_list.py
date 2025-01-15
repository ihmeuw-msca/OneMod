from typing import Hashable, List, TypeVar

from pydantic import AfterValidator
from typing_extensions import Annotated

T = TypeVar("T", bound=Hashable)


def unique_list(items: List[T]) -> List[T]:
    """Ensure all items in the list are unique while preserving order."""
    if items is None:
        return
    return list(dict.fromkeys(items))


UniqueList = Annotated[List[T], AfterValidator(unique_list)]


def update_unique_list(list1: List[T], list2: List[T]) -> List[T]:
    """Combine two lists, remove duplicates, and preserve order."""
    return list(dict.fromkeys(list1 + list2))
