import pytest
from pydantic import ValidationError

from onemod.dtypes.unique_list import UniqueList


@pytest.mark.unit
def test_unique_list_valid():
    # Test a valid list
    unique_list = UniqueList(items=["a", "b", "c"])
    assert unique_list.items == ["a", "b", "c"]


@pytest.mark.unit
def test_unique_list_valid_non_explicit():
    # Test a valid list without explicitly naming the items argument
    unique_list = UniqueList(["a", "b", "c"])
    assert unique_list.items == ["a", "b", "c"]


@pytest.mark.unit
def test_unique_list_empty():
    # Test an empty list
    unique_list = UniqueList(items=[])
    assert unique_list.items == []


@pytest.mark.unit
def test_unique_list_invalid_duplicates():
    # Test a list with duplicate items
    with pytest.raises(ValidationError) as exc_info:
        UniqueList(items=["a", "b", "a"])
    assert "All items in the list must be unique" in str(exc_info.value)


@pytest.mark.unit
def test_unique_list_type_enforcement():
    # Test that input must be a list
    with pytest.raises(TypeError) as exc_info:
        UniqueList(items="not_a_list")  # Invalid input type
    assert "Input must be a list" in str(exc_info.value)


@pytest.mark.unit
def test_unique_list_append():
    # Test appending an item to the list
    unique_list = UniqueList(items=["a", "b"])
    unique_list.append("c")
    assert unique_list.items == ["a", "b", "c"]

    # Test appending a duplicate item
    with pytest.raises(ValueError) as exc_info:
        unique_list.append("a")
    assert "Item a already exists in UniqueList" in str(exc_info.value)


def test_unique_list_iterable():
    # Test that the UniqueList is iterable
    unique_list = UniqueList(items=["x", "y", "z"])
    items = [item for item in unique_list]
    assert items == ["x", "y", "z"]


def test_unique_list_indexing():
    # Test indexing behavior
    unique_list = UniqueList(items=["alpha", "beta", "gamma"])
    assert unique_list[0] == "alpha"
    assert unique_list[1] == "beta"
    assert unique_list[2] == "gamma"


def test_unique_list_len():
    # Test the length of the list
    unique_list = UniqueList(items=["one", "two", "three"])
    assert len(unique_list) == 3


def test_unique_list_model_integration():
    # Test integration with another Pydantic model
    from pydantic import BaseModel

    class TestModel(BaseModel):
        unique_items: UniqueList[str]

    # Valid case
    model = TestModel(unique_items=UniqueList(items=["1", "2", "3"]))
    assert model.unique_items.items == ["1", "2", "3"]

    # Invalid case
    with pytest.raises(ValidationError) as exc_info:
        TestModel(unique_items=UniqueList(items=["1", "1", "2"]))
    assert "All items in the list must be unique" in str(exc_info.value)
