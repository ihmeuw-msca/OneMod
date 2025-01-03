import pytest
from pydantic import BaseModel, ValidationError, validate_call

from onemod.dtypes.unique_list import UniqueList, unique_list


# Sub-class of BaseModel with a UniqueList[int] field
class UniqueListIntModel(BaseModel):
    unique_list: UniqueList[int]


# Sub-class of BaseModel with a UniqueList[str] field
class UniqueListStrModel(BaseModel):
    unique_list: UniqueList[str]


# Function taking a UniqueList[str] argument
@validate_call
def unique_list_func(unique_list: UniqueList[str]):
    return unique_list


# Test the unique_list validator function
@pytest.mark.unit
def test_unique_list_valid():
    assert unique_list(["a", "b", "c"]) == ["a", "b", "c"]  # Valid unique list


@pytest.mark.unit
def test_unique_list_non_unique():
    assert unique_list([1, 2, 2]) == [
        1,
        2,
    ]  # Non-unique items should be removed


@pytest.mark.unit
def test_unique_list_model_valid():
    model = UniqueListIntModel(unique_list=[1, 2, 3])
    assert model.unique_list == [1, 2, 3]  # Ensure valid list is accepted


@pytest.mark.unit
def test_unique_list_model_non_unique():
    model = UniqueListStrModel(unique_list=["a", "b", "b"])
    assert model.unique_list == [
        "a",
        "b",
    ]  # Ensure non-unique items are removed


@pytest.mark.unit
def test_unique_list_model_empty_list():
    model = UniqueListIntModel(unique_list=[])
    assert model.unique_list == []  # Ensure empty lists are allowed


@pytest.mark.unit
def test_unique_list_model_none_as_list():
    with pytest.raises(ValidationError) as exc_info:
        UniqueListIntModel(unique_list=None)  # None is not valid for List[int]
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "list_type"
    assert errors[0]["msg"] == "Input should be a valid list"


@pytest.mark.unit
def test_unique_list_func_valid():
    assert unique_list_func(["a", "b", "c"]) == ["a", "b", "c"]


@pytest.mark.unit
def test_unique_list_func_non_unique():
    assert unique_list_func(["a", "b", "b"]) == ["a", "b"]
