from typing import Any, Dict, Union

from pandas import DataFrame
from pydantic import BaseModel


class Data(BaseModel):
    columns: Dict[str, Any]

    @classmethod
    def with_columns(cls, columns: Dict[str, Union[type, BaseModel]]) -> 'Data':
        """Creates a Data object with specified column types."""
        return cls(columns=columns)

    def validate_columns(self, data: DataFrame) -> None:
        """Validate that each row in the DataFrame matches the specified type for that column."""
        for index, row in data.iterrows():  # TODO: computational overhead worth the benefit of pre-run validation?
            for col_name, col_type in self.columns.items():  # We could just do this loop alone to check dtypes, but we would lose the constraints checking functionality
                if col_name in row:
                    value = row[col_name]
                    if not isinstance(value, col_type):
                        raise ValueError(
                            f"Column '{col_name}' in row {index} expected to be of type {col_type}, but got {type(value)}"
                        )
                else:
                    raise ValueError(f"Missing required column: '{col_name}' in row {index}")
