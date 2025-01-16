"""Helper functions for unique lists and tuples."""

from typing import Annotated, Hashable, TypeVar

from pydantic import AfterValidator

T = TypeVar("T", bound=Hashable)


def unique_list(items: list[T]) -> list[T]:
    """Ensure all items in list are unique while preserving order."""
    return list(dict.fromkeys(items))


def update_unique_list(list1: list[T], list2: list[T]) -> list[T]:
    """Combine two lists, remove duplicates, and preserve order."""
    return list(dict.fromkeys(list1 + list2))


def unique_tuple(items: tuple[T, ...]) -> tuple[T, ...]:
    """Ensure all items in tuple are unique while preserving order."""
    return tuple(dict.fromkeys(items))


def update_unique_tuple(
    tuple1: tuple[T, ...], tuple2: tuple[T, ...]
) -> tuple[T, ...]:
    """Combine two tuples, remove duplicates, and preserve order."""
    return tuple(dict.fromkeys(tuple1 + tuple2))


UniqueList = Annotated[list[T], AfterValidator(unique_list)]
UniqueTuple = Annotated[tuple[T, ...], AfterValidator(unique_tuple)]
