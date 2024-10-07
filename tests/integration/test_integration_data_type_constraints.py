from polars import DataFrame
import pytest

from onemod.constraints import bounds
from onemod.types import Data

@pytest.mark.integration
def test_data_with_integer_with_bounds_valid():
    schema = Data.use_validation(dict(
        age_group_id=dict(
            type=int,
            constraints=[bounds(0, 500)]
        ),
        location_id=dict(
            type=int
        )
    ))

    valid_data = DataFrame({
        "age_group_id": [50, 30, 20],
        "location_id": [1, 2, 3]
    })
    
    schema.validate_columns(valid_data)  # No error should be raised

@pytest.mark.integration
def test_data_with_integer_with_bounds_valid_shape():
    schema = Data.use_validation(
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
    schema = Data.use_validation(
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
        schema.validate_columns(invalid_data)
    assert "Expected DataFrame shape (3, 2)" in str(excinfo.value)
    
@pytest.mark.integration
def test_data_with_constraints_invalid_and_shape_invalid():
    """Ensure that both errors are reported when both are present."""
    schema = Data.use_validation(
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

    with pytest.raises(ValueError) as excinfo:
        schema.validate_columns(invalid_data)
    assert "Expected DataFrame shape (3, 2)" in str(excinfo.value)
    assert "expected to be of type" in str(excinfo.value)
