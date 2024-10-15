
from polars import DataFrame
import pytest

from onemod.types import Data

@pytest.mark.unit
def test_data_with_missing_column_in_dataframe():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id={},
            location_id={}
        )
    )

    invalid_data = DataFrame({
        "age_group_id": [50, 30, 20]
        # Missing 'location_id'
    })

    with pytest.raises(ValueError) as excinfo:
        schema.validate_columns(invalid_data)
    assert "Column 'location_id' is missing from the data" in str(excinfo.value)

@pytest.mark.unit
def test_data_with_extra_column():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id={},
            location_id={}
        )
    )

    extra_data = DataFrame({
        "age_group_id": [50, 30, 20],
        "location_id": [1, 2, 3],
        "extra_column": ["extra", "extra", "extra"]  # Should be ignored
    })

    schema.validate_columns(extra_data)  # No error should be raised
    
@pytest.mark.unit
def test_data_with_integer_valid():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=dict(
                type=int,
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

@pytest.mark.unit
def test_data_with_type_mismatch_in_dataframe():
    schema = Data(
        stage="test",
        path="test.parquet",
        columns=dict(
            age_group_id=dict(
                type=int,
            ),
            location_id=dict(
                type=int
            )
        )
    )

    invalid_data = DataFrame({
        "age_group_id": [50, 30, 20],
        "location_id": ["one", "two", "three"]  # Wrong type
    })

    with pytest.raises(ValueError) as excinfo:
        schema.validate_columns(invalid_data)
    assert "Column 'location_id' must be of type" in str(excinfo.value)
