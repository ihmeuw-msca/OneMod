from typing import Hashable, List, TypeVar

from pydantic import AfterValidator
from typing_extensions import Annotated

T = TypeVar("T", bound=Hashable)


def unique_list(items: List[T]) -> List[T]:
    """Ensure all items in the list are unique while preserving order."""
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


UniqueList = Annotated[List[T], AfterValidator(unique_list)]
