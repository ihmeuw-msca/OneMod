from pydantic import BaseModel, ConfigDict
from typing import Any

class ParametrizedBaseModel(BaseModel):
    """An extension of BaseModel that supports some dict-like dunder methods.

    This model has a base config to allow extra arguments and to validate attribute assignment post init,
    can be overriden after instantiation."""
    model_config = ConfigDict(extra='allow', frozen=False, validate_assignment=True)
    parent_args: dict[str, Any] = {}

    def __getitem__(self, item):
        return getattr(self, item)

    def __contains__(self, key: Any) -> bool:
        return key in self.__dict__

    def __setitem__(self, key: Any, value: Any) -> None:
        setattr(self, key, value)

    def inherit(self, keys: list[str] | None = None) -> None:
        """Inherit the values of the keys from the parent model."""
        if not keys:
            return
        for key in keys:
            if not getattr(self, key, None) and key in self.parent_args:
                setattr(self, key, self.parent_args.get(key))
