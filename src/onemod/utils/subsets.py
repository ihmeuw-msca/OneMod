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


def get_subset(
    data: Path | str,
    subsets: Path | str,
    subset_id: int,
    id_names: list[str] | None = None,
) -> pd.DataFrame:
    """Get data subset by subset_id."""
    id_subsets = get_id_subsets(subsets, subset_id)
    if id_names is not None:
        id_subsets = {
            key: value for key, value in id_subsets.items() if key in id_names
        }
    return filter_data(data, id_subsets)


def get_id_subsets(subsets: Path | str, subset_id: int) -> dict:
    """Get ID names and values that define a data subset."""
    return (
        pd.read_csv(subsets)
        .query("subset_id == @subset_id")
        .drop(columns=["subset_id"])
        .iloc[0]
        .to_dict()
    )


def filter_data(
    data_path: Path | str, id_subsets: dict[str, set[int]]
) -> pd.DataFrame:
    """Filter data by ID subsets."""
    if (suffix := Path(data_path).suffix) == ".csv":
        data = pd.read_csv(data_path)
    elif suffix == ".parquet":
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return data.query(
        " & ".join(
            [f"{key}.isin({value})" for key, value in id_subsets.items()]
        )
    ).reset_index(drop=True)
