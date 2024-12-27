from typing import List

import pytest
from pydantic import AfterValidator, BaseModel, ValidationError
from typing_extensions import Annotated

from onemod.dtypes.validation import is_unique_list


class UniqueListModel(BaseModel):
    unique_list: Annotated[List[int], AfterValidator(is_unique_list)]


@pytest.mark.unit
def test_is_unique_list_passes():
    assert is_unique_list([1, 2, 3]) == [1, 2, 3]  # Valid unique list


@pytest.mark.unit
def test_is_unique_list_raises_value_error():
    with pytest.raises(
        ValueError, match="All items in the list must be unique"
    ):
        is_unique_list([1, 2, 2])  # Non-unique list


@pytest.mark.unit
def test_unique_list_model_passes():
    model = UniqueListModel(unique_list=[1, 2, 3])
    assert model.unique_list == [1, 2, 3]  # Ensure data is passed correctly


@pytest.mark.unit
def test_unique_list_model_fails_validation():
    with pytest.raises(ValidationError) as exc_info:
        UniqueListModel(unique_list=[1, 2, 2])  # Non-unique list
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert (
        errors[0]["msg"] == "Value error, All items in the list must be unique"
    )
    assert errors[0]["type"] == "value_error"


@pytest.mark.unit
def test_unique_list_model_empty_list():
    model = UniqueListModel(unique_list=[])
    assert model.unique_list == []  # Ensure empty lists are allowed


@pytest.mark.unit
def test_unique_list_model_none_as_list():
    with pytest.raises(ValidationError) as exc_info:
        UniqueListModel(unique_list=None)  # None is not valid for List[int]
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "list_type"
    assert errors[0]["msg"] == "Input should be a valid list"
