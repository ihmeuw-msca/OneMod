from polars import DataFrame
import pytest

from onemod.constraints import bounds
from onemod.types import Data

@pytest.mark.integration
def test_data_with_integer_with_bounds_valid():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=dict(
                type=int,
                constraints=[bounds(0, 500)]
            ),
            location_id=dict(
                type=int
            )
        )
    )

    valid_data = DataFrame({
        "age_group_id": [50, 30, 20],
        "location_id": [1, 2, 3]
    })
    
    schema.validate_columns(valid_data)  # No error should be raised

@pytest.mark.integration
def test_data_with_integer_with_bounds_valid_shape():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=dict(
                type=int,
                constraints=[bounds(0, 500)]
            ),
            location_id=dict(
                type=int
            )
        ),
        shape=(3, 2)
    )

    # Valid DataFrame
    valid_data = DataFrame({
        "age_group_id": [50, 30, 20],
        "location_id": [1, 2, 3]
    })
    schema.validate_columns(valid_data)  # No error should be raised

@pytest.mark.integration
def test_data_with_integer_with_bounds_invalid_shape():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=dict(
                type=int,
                constraints=[bounds(0, 500)]
            ),
            location_id=dict(
                type=int
            )
        ),
        shape=(3, 2)
    )
    
    # Invalid shape DataFrame
    invalid_data = DataFrame({
        "age_group_id": [50, 30],
        "location_id": [1, 2]
    })

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
            age_group_id=dict(
                type=int,
                constraints=[bounds(0, 500)]
            ),
            location_id=dict(
                type=int
            )
        ),
        shape=(3, 2)
    )
    
    # Invalid shape and type DataFrame
    invalid_data = DataFrame({
        "age_group_id": [50, 30],
        "location_id": ["one", "two"]
    })

    # Validate the data (both shape and column validation)
    schema.validate_data(invalid_data, validation_collector)
    assert validation_collector.has_errors()
    errors = validation_collector.get_errors()
    assert len(errors) == 2
    assert any("Column 'location_id' must be of type int." in error.message for error in errors)
    assert any("Expected DataFrame shape (3, 2), got (2, 2)." in error.message for error in errors)
