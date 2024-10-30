"""Functions for working with groupby and subsets."""

from pandas import DataFrame


def create_subsets(groupby: set[str], data: DataFrame) -> DataFrame:
    """Create subsets from groupby."""
    groups = data.groupby(list(groupby))
    subsets = DataFrame(
        [subset for subset in groups.groups.keys()],
        columns=groups.keys,  # type: ignore
    )
    subsets["subset_id"] = subsets.index
    return subsets[["subset_id", *groupby]]


def get_subset(
    data: DataFrame,
    subsets: DataFrame,
    subset_id: int,
    id_names: list[str] | None = None,
) -> DataFrame:
    """Get data subset by subset_id."""
    id_subsets = get_id_subsets(subsets, subset_id)
    if id_names is not None:
        id_subsets = {
            key: value for key, value in id_subsets.items() if key in id_names
        }
    return filter_data(data, id_subsets)


def get_id_subsets(subsets: DataFrame, subset_id: int) -> dict:
    """Get ID names and values that define a data subset."""
    return (
        subsets.query("subset_id == @subset_id")
        .drop(columns=["subset_id"])
        .to_dict(orient="list")
    )


def filter_data(data: DataFrame, id_subsets: dict[str, set[int]]) -> DataFrame:
    """Filter data by ID subsets."""
    return data.query(
        " & ".join(
            [f"{key}.isin({value})" for key, value in id_subsets.items()]
        )
    ).reset_index(drop=True)
