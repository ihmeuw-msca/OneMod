# FIXME: Removed subsets module from utils, move to stage integration tests

# import pandas as pd
# import pytest

# from onemod.utils.subsets import create_subsets


# @pytest.fixture
# def sample_data():
#     return {
#         "age_group_id": [1, 2, 2, 3],
#         "location_id": [10, 20, 20, 30],
#         "sex_id": [1, 2, 1, 2],
#         "value": [100, 200, 300, 400],
#     }


# @pytest.fixture
# def sample_scrambled_data():
#     return {
#         "age_group_id": [3, 2, 1, 2, 2],
#         "location_id": [10, 30, 20, 10, 10],
#         "sex_id": [1, 2, 2, 1, 2],
#         "value": [500, 200, 400, 300, 100],
#     }


# @pytest.mark.unit
# def test_create_subsets_valid(sample_data):
#     """Test create_subsets() with valid input."""
#     data = pd.DataFrame(sample_data)
#     groupby = ["age_group_id", "location_id"]
#     subsets = create_subsets(groupby=groupby, data=data)
#     print(subsets)
#     assert subsets.shape == (3, 3)
#     assert subsets.columns.tolist() == [
#         "subset_id",
#         "age_group_id",
#         "location_id",
#     ]
#     assert subsets["age_group_id"].tolist() == [1, 2, 3]
#     assert subsets["location_id"].tolist() == [10, 20, 30]


# @pytest.mark.unit
# def test_create_subsets_valid_scrambled_data(sample_scrambled_data):
#     """Test create_subsets() with valid input."""
#     data = pd.DataFrame(sample_scrambled_data)
#     groupby = ["age_group_id", "location_id"]
#     subsets = create_subsets(groupby=groupby, data=data)
#     print(subsets)
#     assert subsets.shape == (4, 3)
#     assert subsets.columns.tolist() == [
#         "subset_id",
#         "age_group_id",
#         "location_id",
#     ]
#     assert subsets["age_group_id"].tolist() == [1, 2, 2, 3]
#     assert subsets["location_id"].tolist() == [20, 10, 30, 10]


# @pytest.mark.unit
# def test_create_subsets_with_duplicate_groupby(sample_data):
#     """Test create_subsets() with duplicate groupby columns."""
#     data = pd.DataFrame(sample_data)
#     groupby = ["age_group_id", "age_group_id", "location_id"]
#     subsets = create_subsets(groupby=groupby, data=data)
#     print(subsets)
#     assert subsets.shape == (3, 3)
#     assert subsets.columns.tolist() == [
#         "subset_id",
#         "age_group_id",
#         "location_id",
#     ]
#     assert subsets["age_group_id"].tolist() == [1, 2, 3]
