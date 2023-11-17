from typing import Any

from pydantic import BaseModel, ConfigDict


class ParametrizedBaseModel(BaseModel):
    """An extension of BaseModel that supports some dict-like dunder methods.

    This model has a base config to allow extra arguments and to validate attribute assignment
    post init, can be overriden after instantiation."""

    model_config = ConfigDict(extra="allow", frozen=False, validate_assignment=True)
    parent_args: dict[str, Any] = {}

    def get(self, item: Any) -> Any:
        try:
            return self.__getitem__(item)
        except AttributeError:
            return None

    def __getitem__(self, item: Any) -> Any:
        return getattr(self, item)

    def __contains__(self, key: Any) -> bool:
        return key in self.model_fields

    def __setitem__(self, key: Any, value: Any) -> None:
        setattr(self, key, value)

    def inherit(self, keys: list[str] | None = None) -> None:
        """Inherit the values of the keys from the parent model."""
        if not keys:
            return
        for key in keys:
            if not getattr(self, key, None) and key in self.parent_args:
                setattr(self, key, self.parent_args.get(key))

    def model_dump(self, *args, **kwargs) -> dict:
        """Exclude parent_args in model dump by default."""
        exclude_keys = kwargs.pop("exclude", set())
        exclude_keys.add("parent_args")
        return super().model_dump(*args, exclude=exclude_keys, **kwargs)
