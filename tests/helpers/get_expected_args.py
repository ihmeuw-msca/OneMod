import pandas as pd


def get_expected_args() -> dict:
    """Dictionary of the expected arguments for each stage."""
    return {
        "preprocessing": {
            "methods": {"run": ["run"], "fit": ["run"], "predict": None},
            "subsets": None,
            "paramsets": None,
        },
        "covariate_selection": {
            "methods": {"run": ["run", "fit"], "fit": ["fit"], "predict": None},
            "subsets": pd.DataFrame(
                {"sex_id": [1, 1, 2, 2], "age_group_id": [2, 3, 2, 3]}
            ),
            "paramsets": None,
        },
        "global_model": {
            "methods": {
                "run": ["run", "fit", "predict"],
                "fit": ["fit"],
                "predict": ["predict"],
            },
            "subsets": pd.DataFrame({"sex_id": [1, 2]}),
            "paramsets": None,
        },
        "location_model": {
            "methods": {
                "run": ["run", "fit", "predict"],
                "fit": ["fit"],
                "predict": ["predict"],
            },
            "subsets": pd.DataFrame(
                {"sex_id": [1, 1, 2, 2], "location_id": [6, 33, 6, 33]}
            ),
            "paramsets": None,
        },
        "smoothing": {
            "methods": {
                "run": ["run", "fit", "predict"],
                "fit": ["fit"],
                "predict": ["predict"],
            },
            "subsets": pd.DataFrame(
                {"sex_id": [1, 1, 2, 2], "region_id": [5.0, 32.0, 5.0, 32.0]}
            ),
            "paramsets": None,
        },
        "custom_stage": {
            "methods": {
                "run": ["run", "fit", "predict"],
                "fit": ["fit"],
                "predict": ["predict"],
            },
            "subsets": pd.DataFrame(
                {
                    "sex_id": [1, 1, 2, 2],
                    "super_region_id": [4.0, 31.0, 4.0, 31.0],
                }
            ),
            "paramsets": pd.DataFrame({"custom_param": [1, 2]}),
        },
    }
