import json

from polars import DataFrame
import pytest

from onemod.constraints import Constraint
from onemod.dtypes import ColumnSpec, Data


@pytest.mark.integration
def test_data_with_integer_with_bounds_valid():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=ColumnSpec(
                type=int,
                constraints=[
                    Constraint(name="bounds", args=dict(ge=0, le=500))
                ],
            ),
            location_id=ColumnSpec(type=int),
        ),
    )

    valid_data = DataFrame(
        {"age_group_id": [50, 30, 20], "location_id": [1, 2, 3]}
    )

    schema.validate_columns(valid_data)  # No error should be raised


@pytest.mark.integration
def test_data_with_integer_with_bounds_valid_shape():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=ColumnSpec(
                type=int,
                constraints=[
                    Constraint(name="bounds", args=dict(ge=0, le=500))
                ],
            ),
            location_id=ColumnSpec(type=int),
        ),
        shape=(3, 2),
    )

    # Valid DataFrame
    valid_data = DataFrame(
        {"age_group_id": [50, 30, 20], "location_id": [1, 2, 3]}
    )
    schema.validate_columns(valid_data)  # No error should be raised


@pytest.mark.integration
def test_data_with_integer_with_bounds_invalid_shape():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=ColumnSpec(
                type=int,
                constraints=[
                    Constraint(name="bounds", args=dict(ge=0, le=500))
                ],
            ),
            location_id=ColumnSpec(type=int),
        ),
        shape=(3, 2),
    )

    # Invalid shape DataFrame
    invalid_data = DataFrame({"age_group_id": [50, 30], "location_id": [1, 2]})

    with pytest.raises(ValueError) as excinfo:
        schema.validate_shape(invalid_data)
    assert "Expected DataFrame shape (3, 2)" in str(excinfo.value)


@pytest.mark.integration
def test_data_with_constraints_invalid_and_shape_invalid(validation_collector):
    """Ensure that both errors are reported when both are present."""
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=ColumnSpec(
                type=int,
                constraints=[
                    Constraint(name="bounds", args=dict(ge=0, le=500))
                ],
            ),
            location_id=ColumnSpec(type=int),
        ),
        shape=(3, 2),
    )

    # Invalid shape and type DataFrame
    invalid_data = DataFrame(
        {"age_group_id": [50, 30], "location_id": ["one", "two"]}
    )

    # Validate the data (both shape and column validation)
    schema.validate_data(invalid_data, validation_collector)
    assert validation_collector.has_errors()
    errors = validation_collector.get_errors()
    assert len(errors) == 2
    assert any(
        "Column 'location_id' must be of type int." in error.message
        for error in errors
    )
    assert any(
        "Expected DataFrame shape (3, 2), got (2, 2)." in error.message
        for error in errors
    )


@pytest.mark.integration
def test_data_model_no_constraints():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=ColumnSpec(type=int), location_id=ColumnSpec(type=int)
        ),
    )

    expected = {
        "stage": "test",
        "path": "test.parquet",
        "format": "parquet",
        "shape": None,
        "columns": {
            "age_group_id": {"type": "int", "constraints": None},
            "location_id": {"type": "int", "constraints": None},
        },
    }

    actual = schema.model_dump()

    assert actual == expected


@pytest.mark.integration
def test_data_model_with_constraints():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=ColumnSpec(
                type=int,
                constraints=[
                    Constraint(name="bounds", args=dict(ge=0, le=500))
                ],
            ),
            selected_covs=ColumnSpec(
                type=str,
                constraints=[
                    Constraint(
                        name="is_in", args=dict(other=["cov1", "cov2", "cov3"])
                    )
                ],
            ),
        ),
    )

    expected = {
        "stage": "test",
        "path": "test.parquet",
        "format": "parquet",
        "shape": None,
        "columns": {
            "age_group_id": {
                "type": "int",
                "constraints": [
                    {"name": "bounds", "args": {"ge": 0, "le": 500}}
                ],
            },
            "selected_covs": {
                "type": "str",
                "constraints": [
                    {
                        "name": "is_in",
                        "args": {"other": ["cov1", "cov2", "cov3"]},
                    }
                ],
            },
        },
    }

    actual = schema.model_dump()

    assert actual == expected


@pytest.mark.integration
def test_data_to_json_no_constraints(tmp_path):
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=ColumnSpec(type=int), location_id=ColumnSpec(type=int)
        ),
    )

    expected = {
        "stage": "test",
        "path": "test.parquet",
        "format": "parquet",
        "columns": {
            "age_group_id": {"type": "int"},
            "location_id": {"type": "int"},
        },
    }

    filepath = tmp_path / "test.json"
    with filepath.open('w') as f:
        f.write(schema.model_dump_json(indent=4, exclude_none=True))
    with filepath.open('r') as f:
        actual = json.load(f)

    assert actual == expected


@pytest.mark.integration
def test_data_to_json_with_constraints(tmp_path):
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=ColumnSpec(
                type=int,
                constraints=[
                    Constraint(name="bounds", args=dict(ge=0, le=500))
                ],
            ),
            selected_covs=ColumnSpec(
                type=str,
                constraints=[
                    Constraint(
                        name="is_in", args=dict(other=["cov1", "cov2", "cov3"])
                    )
                ],
            ),
        ),
    )

    expected = {
        "stage": "test",
        "path": "test.parquet",
        "format": "parquet",
        "columns": {
            "age_group_id": {
                "type": "int",
                "constraints": [
                    {"name": "bounds", "args": {"ge": 0, "le": 500}}
                ],
            },
            "selected_covs": {
                "type": "str",
                "constraints": [
                    {
                        "name": "is_in",
                        "args": {"other": ["cov1", "cov2", "cov3"]},
                    }
                ],
            },
        },
    }

    filepath = tmp_path / "test.json"
    with filepath.open('w') as f:
        f.write(schema.model_dump_json(indent=4, exclude_none=True))
    with filepath.open('r') as f:
        actual = json.load(f)

    assert actual == expected
