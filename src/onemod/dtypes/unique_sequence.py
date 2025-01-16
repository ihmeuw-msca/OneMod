"""Helper functions for unique lists and tuples.

Notes
-----
There are many `Pipeline` and `Stage` fields that we expect to contain
unique items (e.g., `groupby`, `crossby`, `stages`). We originally
implemented these as sets, but this caused some issues. For example, the
fact that the `groupby` set didn't preserve order caused `subset`
ordering to be unpredictable. We also often had to cast the `groupby`
set as a list before calling pandas functions. The functions and type
annotations here allow us to ensure unique items while preserving order.
Because they are only validators and not actual types, they don't
guarantee uniqueness when fields are modified by the user.

"""

from typing import Hashable, List, Tuple, TypeVar

from pydantic import AfterValidator
from typing_extensions import Annotated

T = TypeVar("T", bound=Hashable)


def unique_list(items: List[T]) -> List[T]:
    """Ensure all items in list are unique while preserving order."""
    return list(dict.fromkeys(items))


def update_unique_list(list1: List[T], list2: List[T]) -> List[T]:
    """Combine two lists, remove duplicates, and preserve order."""
    return list(dict.fromkeys(list1 + list2))


def unique_tuple(items: Tuple[T, ...]) -> Tuple[T, ...]:
    """Ensure all items in tuple are unique while preserving order."""
    return tuple(dict.fromkeys(items))


def update_unique_tuple(
    tuple1: Tuple[T, ...], tuple2: Tuple[T, ...]
) -> Tuple[T, ...]:
    """Combine two tuples, remove duplicates, and preserve order."""
    return tuple(dict.fromkeys(tuple1 + tuple2))


UniqueList = Annotated[List[T], AfterValidator(unique_list)]
UniqueTuple = Annotated[Tuple[T, ...], AfterValidator(unique_tuple)]
