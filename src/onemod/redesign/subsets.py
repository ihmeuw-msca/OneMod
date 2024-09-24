"""Functions for working with groupby and subsets."""

from pathlib import Path

import pandas as pd


def create_subsets(groupby: set[str], data: Path | str) -> pd.DataFrame | None:
    """Create subsets from groupby."""
    if len(groupby) == 0:
        return None
    data = pd.read_parquet(data, columns=groupby)
    groups = data.groupby(list(groupby))
    subsets = pd.DataFrame(
        [subset for subset in groups.groups.keys()], columns=groups.keys
    )
    subsets["subset_id"] = subsets.index
    return subsets[["subset_id", *groupby]]


def filter_subset(
    data: Path | str, subsets: Path | str, subset_id: int
) -> pd.DataFrame:
    """Filter data by subset_id."""
    subset = pd.read_csv(subsets).query("subset_id == @subset_id")
    id_subsets = {key: {value.item()} for key, value in subset.items()}
    return filter_data(data, id_subsets)


def filter_data(
    data: Path | str, id_subsets: dict[str, set[int]]
) -> pd.DataFrame:
    """Filter data by id subsets."""
    return (
        pd.read_parquet(data)
        .query(
            " & ".join(
                [f"{key}.isin({value})" for key, value in id_subsets.items()]
            )
        )
        .reset_index(drop=True)
    )
