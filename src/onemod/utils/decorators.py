from pydantic import computed_field


def computed_property(func):
    """Combines @computed_field and @property."""
    return computed_field(property(func))
