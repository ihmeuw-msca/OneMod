from collections.abc import Collection
from typing import Any

from polars import Series

def bounds(ge: float = None, le: float = None) -> callable:
    """
    Returns a function that checks if all values in the column are within the specified bounds.
    
    Parameters
    ----------
    ge (float): Minimum allowed value (inclusive)
    le (float): Maximum allowed value (inclusive)
    
    Examples
    --------
    >>> validate = bounds(ge=0, le=100)
    >>> validate(pl.Series([1, 2, 3, 4, 5]))  # No error
    >>> validate(pl.Series([-1, 2, 3, 4, 5]))  # ValueError: All values must be greater than or equal to 0.
    
    Returns
    -------
        callable: Function that validates the column values
    """
    
    def validate(column: Series) -> None:
        if ge is not None:
            if not column.ge(ge).all():
                raise ValueError(f"All values must be greater than or equal to {ge}.")
        
        if le is not None:
            if not column.le(le).all():
                raise ValueError(f"All values must be less than or equal to {le}.")
    
    return validate

def is_in(other: Collection[Any]) -> callable:
    """
    Returns a function that checks if all values in the column are within the specified collection.
    
    Parameters
    ----------
    other (Collection[Any]): Collection of values
    
    Examples
    --------
    >>> validate = is_in(["a", "b", "c"])
    >>> validate(pl.Series(["a", "b"]))  # No error
    >>> validate(pl.Series(["a", "d"]))  # ValueError: All values must be in ['a', 'b', 'c'].
    
    Returns
    -------
        callable: Function that validates the column values
    """
    
    def validate(column: Series) -> None:
        if not column.is_in(other).all():
            raise ValueError(f"All values must be in {other}.")
    
    return validate

def no_inf() -> callable:
    """
    Returns a function that checks that there are no infinite(inf, -inf) values in the column.
    
    Examples
    --------
    >>> validate = no_inf()
    >>> validate(pl.Series([1, 2, 3, 4, 5]))  # No error
    >>> validate(pl.Series([1, 2, float('inf'), 4, 5]))  # ValueError: All values must be finite.
    
    Returns
    -------
        callable: Function that validates the column values
    """
    
    def validate(column: Series) -> None:
        if column.is_in([float('inf'), float('-inf')]).any():
            raise ValueError("All values must be finite.")
    
    return validate
