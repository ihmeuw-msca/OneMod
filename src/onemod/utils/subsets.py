"""Functions for working with groupby and subsets."""

from typing import List

import pandas as pd
from pydantic import ConfigDict, validate_call

from onemod.dtypes import UniqueList


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def create_subsets(
    groupby: UniqueList[str], data: pd.DataFrame
) -> pd.DataFrame:
    """Create subsets from groupby."""
    sorted_groupby = sorted(groupby)
    groups = data.groupby(sorted_groupby)
    subsets = pd.DataFrame(
        [subset for subset in groups.groups.keys()], columns=sorted_groupby
    )
    subsets.sort_values(by=sorted_groupby)
    subsets["subset_id"] = subsets.index
    return subsets[["subset_id", *sorted_groupby]]


def get_subset(
    data: pd.DataFrame,
    subsets: pd.DataFrame,
    subset_id: int,
    id_names: List[str] | None = None,
) -> pd.DataFrame:
    """Get data subset by subset_id."""
    id_subsets = get_id_subsets(subsets, subset_id)
    if id_names is not None:
        id_subsets = {
            key: value for key, value in id_subsets.items() if key in id_names
        }
    return filter_data(data, id_subsets)


def get_id_subsets(subsets: pd.DataFrame, subset_id: int) -> dict:
    """Get ID names and values that define a data subset."""
    return (
        subsets.query("subset_id == @subset_id")
        .drop(columns=["subset_id"])
        .to_dict(orient="list")
    )


def filter_data(
    data: pd.DataFrame, id_subsets: dict[str, set[int]]
) -> pd.DataFrame:
    """Filter data by ID subsets."""
    return data.query(
        " & ".join(
            [f"{key}.isin({value})" for key, value in id_subsets.items()]
        )
    ).reset_index(drop=True)
