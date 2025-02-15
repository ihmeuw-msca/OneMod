from collections.abc import Collection
from typing import Any, Callable

from polars import Series


def bounds(
    ge: float | None = None, le: float | None = None
) -> Callable[[Series], None]:
    """
    Returns a function that checks if all values in the column are within the specified bounds.

    Parameters
    ----------
    ge (float): Minimum allowed value (inclusive)
    le (float): Maximum allowed value (inclusive)

    Examples
    --------
    >>> use_validation = bounds(ge=0, le=100)
    >>> use_validation(pl.Series([1, 2, 3, 4, 5]))  # No error
    >>> use_validation(
    ...     pl.Series([-1, 2, 3, 4, 5])
    ... )  # ValueError: All values must be greater than or equal to 0.

    Returns
    -------
        callable: Function that validates the column values
    """

    def use_validation(column: Series) -> None:
        if ge is not None:
            if not column.ge(ge).all():
                raise ValueError(
                    f"All values must be greater than or equal to {ge}."
                )

        if le is not None:
            if not column.le(le).all():
                raise ValueError(
                    f"All values must be less than or equal to {le}."
                )

    return use_validation


def is_in(other: Collection[Any]) -> Callable[[Series], None]:
    """
    Returns a function that checks if all values in the column are within the specified collection.

    Parameters
    ----------
    other (Collection[Any]): Collection of values

    Examples
    --------
    >>> use_validation = is_in(["a", "b", "c"])
    >>> use_validation(pl.Series(["a", "b"]))  # No error
    >>> use_validation(
    ...     pl.Series(["a", "d"])
    ... )  # ValueError: All values must be in ['a', 'b', 'c'].

    Returns
    -------
        callable: Function that validates the column values
    """

    def use_validation(column: Series) -> None:
        if not column.is_in(other).all():
            raise ValueError(f"All values must be in {other}.")

    return use_validation


def no_inf() -> Callable[[Series], None]:
    """
    Returns a function that checks that there are no infinite(inf, -inf) values in the column.

    Examples
    --------
    >>> use_validation = no_inf()
    >>> use_validation(pl.Series([1, 2, 3, 4, 5]))  # No error
    >>> use_validation(
    ...     pl.Series([1, 2, float("inf"), 4, 5])
    ... )  # ValueError: All values must be finite.

    Returns
    -------
        callable: Function that validates the column values
    """

    def use_validation(column: Series) -> None:
        if column.is_in([float("inf"), float("-inf")]).any():
            raise ValueError("All values must be finite.")

    return use_validation
