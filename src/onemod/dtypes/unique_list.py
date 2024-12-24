from typing import Generic, List, TypeVar

from pydantic import BaseModel, model_validator

T = TypeVar("T")


class UniqueList(BaseModel, Generic[T]):
    """
    A custom Pydantic-compatible type that enforces uniqueness of items in a list.
    """

    items: List[T]

    @model_validator(mode="before")
    @classmethod
    def check_uniqueness(cls, value: List[T]) -> dict:
        if isinstance(value, dict) and "items" in value:
            value = value["items"]
        if not isinstance(value, list):
            raise TypeError("Input must be a list")
        if len(value) != len(set(value)):
            raise ValueError("All items in the list must be unique")
        return {"items": value}

    def __init__(self, items=None, **kwargs):
        if items is not None:
            kwargs["items"] = items
        super().__init__(**kwargs)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def append(self, item: T):
        if item in self.items:
            raise ValueError(f"Item {item} already exists in UniqueList")
        self.items.append(item)
