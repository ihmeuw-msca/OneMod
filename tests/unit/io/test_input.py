"""Test input class."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from onemod.dtypes import Data
from onemod.io import Input

REQUIRED_INPUT = {
    "data": {"format": "parquet"},
    "covariates": {"format": "csv"},
}
OPTIONAL_INPUT = {"priors": {"format": "pkl"}}
VALID_ITEMS = {
    "data": "/path/to/predictions.parquet",
    "covariates": Data(stage="first_stage", path="/path/to/selected_covs.csv"),
    "priors": Data(stage="second_stage", path="/path/to/model.pkl"),
}
ITEMS_WITH_CYCLES = {
    "data": "/path/to/predictions.parquet",
    "covariates": Data(stage="test_stage", path="/path/to/selected_covs.csv"),
    "priors": Data(stage="test_stage", path="/path/to/model.pkl"),
}
ITEMS_WITH_INVALID_TYPES = {
    "data": "/path/to/predictions.csv",
    "covariates": Data(
        stage="first_stage", path="/path/to/selected_covs.parquet"
    ),
    "priors": Data(stage="second_stage", path="/path/to/model.zip"),
}
ITEMS_WITH_EXTRAS = {"dummy": "/path/to/dummy.parquet", **VALID_ITEMS}
ITEMS_WITH_SHARED_DEPENDENCY = {
    "data": "/path/to/predictions.parquet",
    "covariates": Data(stage="first_stage", path="/path/to/selected_covs.csv"),
    "priors": Data(stage="first_stage", path="/path/to/model.pkl"),
}


def get_input(items: dict[str, Data | Path | str] = {}) -> Input:
    if items:
        items = items.copy()
        for item_name in items:
            if isinstance(item_value := items[item_name], (Path, str)):
                items[item_name] = Input.path_to_data(item_value)

    return Input(
        stage="test_stage",
        required=REQUIRED_INPUT,
        optional=OPTIONAL_INPUT,
        items=items,
    )


@pytest.mark.unit
def test_cycle_detected_by_init():
    with pytest.raises(ValueError):
        get_input(ITEMS_WITH_CYCLES)


@pytest.mark.unit
def test_cycle_detected_by_setitem():
    test_input = get_input()
    with pytest.raises(ValueError) as error:
        test_input["covariates"] = ITEMS_WITH_CYCLES["covariates"]
    assert (
        str(error.value)
        == f"Circular dependency for {test_input.stage} input: covariates"
    )
    assert test_input.items == {}


@pytest.mark.unit
def test_cycle_detected_by_update():
    test_input = get_input()
    with pytest.raises(ValueError) as error:
        test_input.update(ITEMS_WITH_CYCLES)
    assert (
        str(error.value)
        == f"Circular dependencies for {test_input.stage} input: ['covariates', 'priors']"
    )
    assert test_input.items == {}


@pytest.mark.unit
def test_invalid_types_detected_by_init():
    with pytest.raises(TypeError):
        get_input(ITEMS_WITH_INVALID_TYPES)


@pytest.mark.unit
@pytest.mark.parametrize("input_name", ["data", "covariates"])
def test_invalid_type_detected_by_setitem(input_name):
    test_input = get_input()
    with pytest.raises(TypeError) as error:
        test_input[input_name] = ITEMS_WITH_INVALID_TYPES[input_name]
    assert (
        str(error.value)
        == f"Invalid type for {test_input.stage} input: {input_name}"
    )
    assert test_input.items == {}


@pytest.mark.unit
def test_invalid_types_detected_by_update():
    test_input = get_input()
    with pytest.raises(TypeError) as error:
        test_input.update(ITEMS_WITH_INVALID_TYPES)
    assert (
        str(error.value)
        == f"Invalid types for {test_input.stage} input: ['data', 'covariates', 'priors']"
    )
    assert test_input.items == {}


@pytest.mark.unit
def test_items_from_init():
    test_input = get_input(VALID_ITEMS)
    for item_name, item_value in VALID_ITEMS.items():
        if isinstance(item_value, (Path, str)):
            item_value = Input.path_to_data(item_value)
        assert test_input[item_name] == item_value


@pytest.mark.unit
@pytest.mark.parametrize("item_name", ["data", "covariates", "priors"])
def test_item_from_setitem(item_name):
    test_input = get_input()
    test_input[item_name] = (item_value := VALID_ITEMS[item_name])
    assert item_name in test_input
    if isinstance(item_value, (Path, str)):
        assert test_input[item_name] == Input.path_to_data(item_value)
    else:
        assert test_input[item_name] == item_value


@pytest.mark.unit
def test_items_from_update():
    test_input = get_input()
    test_input.update(VALID_ITEMS)
    for item_name, item_value in VALID_ITEMS.items():
        if isinstance(item_value, (Path, str)):
            item_value = Input.path_to_data(item_value)
        assert test_input[item_name] == item_value


@pytest.mark.unit
def test_extras_ignored_by_init():
    test_input = get_input(ITEMS_WITH_EXTRAS)
    assert "dummy" not in test_input


@pytest.mark.unit
def test_extra_ignored_by_setitem():
    test_input = get_input()
    test_input["dummy"] = ITEMS_WITH_EXTRAS["dummy"]
    assert "dummy" not in test_input


@pytest.mark.unit
def test_extras_ignored_by_update():
    test_input = get_input()
    test_input.update(ITEMS_WITH_EXTRAS)
    assert "dummy" not in test_input


@pytest.mark.unit
def test_get():
    test_input = get_input(VALID_ITEMS)
    for input_name, input_value in VALID_ITEMS.items():
        if isinstance(input_value, (Path, str)):
            input_value = Input.path_to_data(input_value)
        assert test_input.get(input_name) == input_value
    assert test_input.get("dummy") is None
    assert test_input.get("dummy", "default") == "default"


@pytest.mark.unit
def test_getitem():
    test_input = get_input(VALID_ITEMS)
    for input_name, input_value in VALID_ITEMS.items():
        if isinstance(input_value, (Path, str)):
            input_value = Input.path_to_data(input_value)
        assert test_input[input_name] == input_value


@pytest.mark.unit
def test_getitem_not_set():
    test_input = get_input()
    for input_name in VALID_ITEMS:
        with pytest.raises(ValueError) as error:
            test_input[input_name]
        assert (
            str(error.value)
            == f"{test_input.stage} input '{input_name}' has not been set"
        )


@pytest.mark.unit
def test_getitem_not_contain():
    test_input = get_input()
    with pytest.raises(KeyError) as error:
        test_input["dummy"]
    assert (
        str(error.value).strip('"')
        == f"{test_input.stage} does not contain input 'dummy'"
    )


@pytest.mark.unit
def test_contains():
    test_input = get_input(VALID_ITEMS)
    for input_name in VALID_ITEMS:
        assert input_name in test_input
    assert "dummy" not in test_input


@pytest.mark.unit
def test_dependencies():
    test_input = get_input(VALID_ITEMS)
    assert test_input.dependencies == ["first_stage", "second_stage"]


@pytest.mark.unit
def test_dependencies_with_duplicate_dependencies():
    test_input = get_input(ITEMS_WITH_SHARED_DEPENDENCY)
    assert test_input.dependencies == ["first_stage"]


@pytest.mark.unit
def test_no_dependencies():
    test_input = get_input(items={"data": "/path/to/predictions.parquet"})
    assert test_input.dependencies == []


@pytest.mark.unit
def test_missing_self():
    test_input = get_input()
    with pytest.raises(KeyError) as error:
        test_input.check_missing()
    observed = str(error.value).strip('"')
    expected = f"Stage '{test_input.stage}' missing required input: "
    assert observed == expected + "['data', 'covariates']"


@pytest.mark.unit
def test_missing_items():
    test_input = get_input()
    with pytest.raises(KeyError) as error:
        test_input.check_missing(
            {
                "priors": Data(
                    stage="second_stage", path=Path("/path/to/model.pkl")
                )
            }
        )
    observed = str(error.value).strip('"')
    expected = f"Stage '{test_input.stage}' missing required input: "
    assert observed == expected + "['data', 'covariates']"


# TODO: Write tests for input.check_exists()
# - input with empty items
# - input with nonemtpy items that don't exist
# - ignore missing/nonexisting item if not in item_names
# - ignore missing/nonexisting item if dependency not in upstream_stages
# - upstream_stages arg not used if item_names passed


@pytest.mark.unit
def test_serialize():
    test_input = get_input(VALID_ITEMS)
    assert test_input.model_dump() == {
        "data": {
            "format": "parquet",
            "path": Path("/path/to/predictions.parquet"),
        },
        "covariates": {
            "stage": "first_stage",
            "format": "csv",
            "path": Path("/path/to/selected_covs.csv"),
        },
        "priors": {
            "stage": "second_stage",
            "format": "pkl",
            "path": Path("/path/to/model.pkl"),
        },
    }
    assert get_input().model_dump() == {}


@pytest.mark.unit
def test_remove():
    test_input = get_input(VALID_ITEMS)
    for item_name in VALID_ITEMS:
        test_input.remove(item_name)
        assert item_name not in test_input


@pytest.mark.unit
def test_clear():
    test_input = get_input(VALID_ITEMS)
    test_input.clear()
    assert test_input.items == {}


@pytest.mark.unit
def test_frozen():
    with pytest.raises(ValidationError):
        get_input().items = VALID_ITEMS
