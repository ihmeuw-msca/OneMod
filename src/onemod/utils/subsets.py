"""Functions for working with groupby and subsets."""

import pandas as pd
import polars as pl


# def create_subsets(groupby: set[str], lf: pl.LazyFrame) -> pl.DataFrame:
#     """Create subsets from groupby."""
#     subsets = (
#         lf.select(list(groupby))
#         .unique()
#         .with_row_index(name="subset_id")
#         .collect()
#     )

#     return subsets


def create_subsets(groupby: set[str], data: pd.DataFrame) -> pd.DataFrame:
    """Create subsets from groupby."""
    sorted_groupby = sorted(groupby)
    groups = data.groupby(sorted_groupby)
    subsets = pd.DataFrame(
        [subset for subset in groups.groups.keys()], columns=groups.keys
    )
    subsets.sort_values(by=sorted_groupby)
    subsets["subset_id"] = subsets.index
    return subsets[["subset_id", *sorted_groupby]]


def get_subset(
    data: pl.DataFrame,
    subsets: pl.DataFrame,
    subset_id: int,
    id_names: list[str] | None = None,
) -> pl.DataFrame:
    """Get data subset by subset_id."""
    id_subsets = get_id_subsets(subsets, subset_id)
    if id_names is not None:
        id_subsets = {
            key: value for key, value in id_subsets.items() if key in id_names
        }
    return filter_data(data, id_subsets)


def get_id_subsets(subsets: pl.DataFrame, subset_id: int) -> dict:
    """Get ID names and values that define a data subset."""
    return (
        subsets.filter(pl.col("subset_id") == subset_id)
        .drop("subset_id")
        .to_dict(as_series=False)
    )


def filter_data(
    data: pl.DataFrame, id_subsets: dict[str, set[int]]
) -> pl.DataFrame:
    """Filter data by ID subsets."""
    filter_expr = pl.lit(True)
    for key, value in id_subsets.items():
        filter_expr &= pl.col(key).is_in(value)

    return data.filter(filter_expr)
