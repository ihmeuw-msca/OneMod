from typing import List


def is_unique_list(items: List) -> List:
    """Check if all items in a list are unique."""
    items = items or []
    if len(items) != len(set(items)):
        raise ValueError("All items in the list must be unique")
    return items
