import pandas as pd
import pytest

from onemod.types.data import Data
from onemod.types.integer import Integer

def test_data_with_valid_dataframe():
    schema = Data.with_columns({
        "age_group_id": Integer.with_bounds(0, 500),
        "location_id": Integer
    })

    valid_data = pd.DataFrame({
        "age_group_id": [50, 30, 20],
        "location_id": [1, 2, 3]
    })
    
    schema.validate_columns(valid_data)  # No error should be raised

def test_data_with_missing_column_in_dataframe():
    schema = Data.with_columns({
        "age_group_id": Integer.with_bounds(0, 500),
        "location_id": Integer
    })

    invalid_data = pd.DataFrame({
        "age_group_id": [50, 30, 20]
        # Missing 'location_id'
    })

    with pytest.raises(ValueError) as excinfo:
        schema.validate_columns(invalid_data)
    assert "Missing required column" in str(excinfo.value)

def test_data_with_type_mismatch_in_dataframe():
    schema = Data.with_columns({
        "age_group_id": Integer.with_bounds(0, 500),
        "location_id": Integer
    })

    invalid_data = pd.DataFrame({
        "age_group_id": [50, 30, 20],
        "location_id": ["one", "two", "three"]  # Wrong type
    })

    with pytest.raises(ValueError) as excinfo:
        schema.validate_columns(invalid_data)
    assert "expected to be of type" in str(excinfo.value)

def test_data_with_multiple_rows_and_extra_columns():
    schema = Data.with_columns({
        "age_group_id": Integer.with_bounds(0, 100),
        "location_id": Integer
    })

    extra_data = pd.DataFrame({
        "age_group_id": [50, 30, 20],
        "location_id": [1, 2, 3],
        "extra_column": ["extra", "extra", "extra"]  # Should be ignored
    })

    schema.validate_columns(extra_data)  # No error should be raised
