"""Useful functions."""
from __future__ import annotations

from functools import cache
from itertools import product
from pathlib import Path
from typing import Any, Optional, Union
import warnings

import numpy as np
import pandas as pd
from pplkit.data.interface import DataInterface
import yaml

from onemod.schema.models.onemod_config import OneModConfig as OneModCFG


class Parameters:
    """Helper class for creating smoother parameter sets.

    In the settings dictionary, some smoother model parameters can be
    specified as either a single item or a list. This class generates
    the cartesian product of all possible parameter combinations and
    assigns each a unique parameter ID.

    Attributes
    ----------
    model_id : str
        Smoother model ID.
    params : tuple of str
        Parameters that can vary by submodel.
    param_sets : pandas.DataFrame
        Parameter set data frame.

    """

    params: tuple[str, ...] = ()

    def __init__(
        self,
        model_id: str,
        config: OneModCFG | None = None,
        param_sets: pd.DataFrame | None = None,
    ) -> None:
        """Create Parameters object.

        Create a Parameters object given either a model settings
        dictionary or a parameter set data frame.

        Parameters
        ----------
        model_id : str
            Smoother model ID.
        config, optional
            Model specifications.
        param_sets : pandas.DataFrame, optional
            Parameter set data frame.

        """
        self.model_id = model_id
        if param_sets is None:
            if config is None:
                raise TypeError("Settings cannot be None if param_sets is None.")
            self.param_sets = self._create_param_sets(config)
        else:
            self.param_sets = param_sets[param_sets["model_id"] == model_id]

    def _create_param_sets(self, config: OneModCFG) -> pd.DataFrame:
        """Create parameter set data frame.

        Parameter set data frame contains parameter IDs and their
        corresponding parameters. Model parameters not included in
        self.params are not included in the data frame as they do not
        vary by submodel.

        Parameters
        ----------
        config
            Model specifications.

        Returns
        -------
        pandas.DataFrame
            Parameter set data frame.

        """
        raise NotImplementedError()

    def get_param_ids(self) -> list:
        """Get list of parameter IDs."""
        return self.param_sets["param_id"].tolist()

    def get_param(self, param: str, param_id: Union[int, str]) -> Any:
        """Get submodel parameter."""
        params = self.param_sets[self.param_sets["param_id"] == param_id]
        return params[param].item()


class SwimrParams(Parameters):
    """Helper class for creating swimr parameter sets."""

    params: tuple[str, ...] = (
        "n_internal_knots_year",
        "similarity_matrix",
        "similarity_multiplier",
        "use_similarity_matrix",
        "theta",
        "intercept_theta",
    )

    def _create_param_sets(self, config: OneModCFG) -> pd.DataFrame:
        """Create parameter set data frame."""
        swimr_config = config["swimr"]
        index = pd.MultiIndex.from_product(
            iterables=[as_list(swimr_config[param]) for param in self.params],
            names=self.params,
        )
        param_sets = pd.DataFrame(index=index).reset_index()
        for param in ["similarity_matrix", "similarity_multiplier"]:
            param_sets.loc[
                param_sets["use_similarity_matrix"] == 0, param
            ] = param_sets[param].unique()[0]
        param_sets.drop_duplicates(inplace=True, ignore_index=True)
        param_cols = list(param_sets.columns)
        param_sets["model_id"] = self.model_id
        param_sets["param_id"] = param_sets.index
        return param_sets[["model_id", "param_id"] + param_cols]


class WeaveParams(Parameters):
    """Helper class for creating weave parameter sets."""

    params: tuple[str, ...] = ("radius", "exponent", "distance_dict")

    def _create_param_sets(self, config: OneModCFG) -> pd.DataFrame:
        """Create parameter set data frame."""
        weave_config = config["weave"]["models"][self.model_id]
        dimensions = weave_config["dimensions"]
        index = pd.MultiIndex.from_product(
            iterables=[
                as_list(dimensions[dimension][param])
                for dimension in dimensions
                for param in self.params
                if param in dimensions[dimension]
            ],
            names=[
                f"{dimension}_{param}"
                for dimension in dimensions
                for param in self.params
                if param in dimensions[dimension]
            ],
        )
        param_sets = pd.DataFrame(index=index).reset_index()
        param_cols = list(param_sets.columns)
        param_sets["model_id"] = self.model_id
        param_sets["param_id"] = param_sets.index
        return param_sets[["model_id", "param_id"] + param_cols]


class Subsets:
    """Helper class for model subsets based on groupby setting.

    Attributes
    ----------
    model_id : str
        Model ID.
    columns : list of str
        Columns used to group data subsets.
    subsets : pandas.DataFrame
        Subset data frame.

    """

    def __init__(
        self,
        model_id: str,
        config: OneModCFG,
        data: pd.DataFrame | None = None,
        subsets: pd.DataFrame | None = None,
    ) -> None:
        """Create Subsets object.

        Parameters
        ----------
        model_id : str
            Model ID.
        config
            Model specifications.
        data : pandas.DataFrame, optional
            Input data.
        subsets : pandas.DataFrame, optional
            Subset data frame.

        """

        self.model_id = model_id
        self.columns = config.groupby
        if subsets is None:
            if data is None:
                raise TypeError("Data cannot be None if subsets are not provided.")
            self.columns = config.groupby
            max_batch_size = config.max_batch
            self.subsets = self._create_subsets(data, max_batch_size)
        else:
            self.subsets = subsets[subsets["model_id"] == model_id]

    def _create_subsets(self, data: pd.DataFrame, max_batch: int) -> pd.DataFrame:
        """Create subset data frame.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data.
        max_batch : int
            Maximum number of prediction points per subset. If -1, do
            not split data into batches.

        Returns
        -------
        pandas.DataFrame
            Subset data frame.

        """
        subsets: dict[str, list] = {column: [] for column in self.columns + ["n_batch"]}
        # TODO: use a different name distinguish column name and column value
        # to improve readibility

        groupby_cols = self.columns
        if len(groupby_cols) == 1:
            groupby_cols = groupby_cols[0]

        for columns, df in data.groupby(groupby_cols):
            if len(self.columns) == 1:
                subsets[self.columns[0]].append(columns)
            else:
                for idx, column in enumerate(self.columns):
                    subsets[column].append(columns[idx])
            n_batch = 1 if max_batch == -1 else int(np.ceil(len(df) / max_batch))
            subsets["n_batch"].append(n_batch)
        subsets_df: pd.DataFrame = pd.DataFrame(subsets)
        subsets_df["model_id"] = self.model_id
        subsets_df["subset_id"] = subsets_df.index
        return subsets_df[["model_id", "subset_id", "n_batch"] + self.columns]

    def get_subset_ids(self) -> list:
        """Get list of subset IDs."""
        return self.subsets["subset_id"].tolist()

    def get_batch_ids(self, subset_id: int) -> list:
        """Get list of batch IDs."""
        n_batch = self.subsets[self.subsets["subset_id"] == subset_id]["n_batch"].item()
        return np.arange(n_batch).tolist()

    def get_column(self, column: str, subset_id: int) -> Any:
        """Get subset column value."""
        subset = self.subsets[self.subsets["subset_id"] == subset_id]
        return subset[column].item()

    def filter_subset(
        self, data: pd.DataFrame, subset_id: int, batch_id: int = 0
    ) -> pd.DataFrame:
        """Filter data by subset and batch."""
        for column in self.columns:
            data = data[data[column] == self.get_column(column, subset_id)]
        n_batch = self.get_column("n_batch", subset_id)
        max_batch = int(np.ceil(len(data) / n_batch))
        batch_start = batch_id * max_batch
        batch_stop = (batch_id + 1) * max_batch
        data["batch"] = False
        data.iloc[batch_start : min(batch_stop, len(data)), -1] = True
        return data


def as_list(values: Union[Any, list]) -> list:
    """Cast values as list if not already."""
    if isinstance(values, (list, tuple, set, dict, np.ndarray)):
        return list(values)
    return [values]


def add_holdouts(
    df: pd.DataFrame,
    n_holdout: int,
    p_holdout: float,
    column: Optional[str] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Add holdout sets to data frame.

    Holdout sets mimic the pattern of missing data in the original data
    set. Code inspired by the CODEm project module
    codem/data/knockouts.py (commit 9ec52be7e7c). Description of method
    and patterns of missing data:

    .. [1] Foreman, K.J., Lozano, R., Lopez, A.D. et al. "Modeling
    causes of death: an integrated approach using CODEm," Popul Health
    Metrics, vol. 10, no. 1, pp. 1-23, 2021.

    Parameters
    -----------
    df : pandas.DataFrame
        Input data.
    n_holdout : int
        Number of holdout sets to create.
    p_holdout : float in (0, 1)
        Percent to holdout for each set.
    column : str, optional
        Name of column with missing data.
    seed : int, optional
        Seed for random number generator.

    Notes
    -----
    * If column is None, assumes missing points already removed from df.
      Otherwise, missing points denoted as NaN values in column.
    * Currently does NOT add random holdouts if less than p_holdout.

    """
    if seed is not None:
        np.random.seed(seed)
    locations1 = df["location_id"].unique()
    locations2 = df["location_id"].unique()
    for n in range(n_holdout):
        c_holdout = 0
        holdout = f"holdout{n+1}"
        df_list = []
        np.random.shuffle(locations1)
        np.random.shuffle(locations2)
        for location1, location2 in zip(locations1, locations2):
            df1 = df[df["location_id"] == location1]
            if column is not None:
                df1 = df1[~df1[column].isna()]
            if c_holdout / len(df) < p_holdout:
                df2 = df[df["location_id"] == location2]
                if column is not None:
                    df2 = df2[~df2[column].isna()]
                df2[holdout] = 0
                df1 = df1.merge(
                    right=df2[["age_group_id", "sex_id", "year_id", holdout]],
                    on=["age_group_id", "sex_id", "year_id"],
                    how="left",
                )
                df1.loc[df1[holdout].isna(), holdout] = 1
                c_holdout += df1[holdout].sum()
            else:
                df1[holdout] = 0
            df_list.append(df1)
        if c_holdout / len(df) < p_holdout:
            p_heldout = c_holdout / len(df)
            warnings.warn(f"Percent held out {p_heldout:.2f} less than {p_holdout}")
        df = df.merge(
            right=pd.concat(df_list)[
                ["age_group_id", "location_id", "sex_id", "year_id", holdout]
            ],
            on=["age_group_id", "location_id", "sex_id", "year_id"],
            how="left",
        )
    return df


def load_settings(
    settings_file: Union[Path, str], raise_on_error: bool = True, as_model: bool = True
) -> OneModCFG | dict:
    """Load settings file."""
    try:
        with open(settings_file, "r") as f:
            settings = yaml.full_load(f)
    except FileNotFoundError:
        if not raise_on_error:
            warnings.warn("Settings file not found; using a null dictionary")
            settings = {}
        else:
            raise
    if not as_model:
        # Return a raw dict, like for task template resources
        return settings
    return OneModCFG(**settings)


def get_rover_covsel_input(config: OneModCFG) -> pd.DataFrame:
    """Get input data for rover model."""
    interface = DataInterface(data=config.input_path)
    df_input = interface.load_data()
    for dimension in config.col_id:
        if dimension in config.id_subsets:
            # Optionally filter data. Defaults to using all values in the dimension
            df_input = df_input[df_input[dimension].isin(config.id_subsets[dimension])]
    return df_input


def get_smoother_input(
    smoother: str,
    config: OneModCFG,
    dataif: DataInterface,
    from_rover: Optional[bool] = False,
) -> pd.DataFrame:
    """Get input data for smoother model."""
    if from_rover:
        df_input = dataif.load_regmod_smooth("predictions.parquet")
        df_input = df_input.rename(columns={"residual": "residual_value"})
    else:
        df_input = get_rover_covsel_input(config)
    columns = _get_smoother_columns(smoother, config).difference(df_input.columns)
    columns = as_list(config.col_id) + list(columns)
    # Deduplicate
    columns = list(set(columns))
    if len(columns) > 0:
        right = dataif.load_data()
        df_input = df_input.merge(
            right=right[columns].drop_duplicates(),
            on=config.col_id,
        )
    if smoother == "weave":  # weave models can't have NaN data
        df_input.loc[df_input[config.col_obs].isna(), "residual_value"] = 1
        df_input.loc[df_input[config.col_obs].isna(), "residual_se"] = 1
    if smoother == "swimr":
        df_input["submodel_id"] = df_input["location_id"].astype(str)
        df_input["row_id"] = np.arange(len(df_input))
        df_input["imputed"] = 0
    return df_input


# TODO: This need to be adjusted for the new change
def _get_smoother_columns(smoother: str, config: OneModCFG) -> set:
    """Get column names needed for smoother model.

    Notes
    -----
    In swimr package function standardize_dataset() from module functions_dataprep.R, it
    seems that the columns super_region_id and region_id will only be retained if both
    are present in the input data. While we may not need both columns for the cascade
    model, we need to make sure both are present if one is needed (e.g., if either
    column is included in the groupby or cascade_level settings):

    if (keep_region_variables & all(c("super_region_id", "region_id") %in% names(df_full)) ) {
      keep_variables <- c("super_region_id", "region_id")
      analytic_vars <- c(analytic_vars, keep_variables)
      new_varnames <- c(new_varnames, keep_variables)
    }

    """
    columns = set(as_list(config.col_holdout) + as_list(config.col_test))
    if smoother not in ("swimr", "weave"):
        raise ValueError(f"Invalid smoother name: {smoother}")
    for model_settings in config[smoother]["models"].values():
        columns.update(as_list(model_settings["groupby"]))
        if smoother == "swimr" and model_settings["model_type"] == "cascade":
            if "cascade_levels" in model_settings:
                columns.update(as_list(model_settings["cascade_levels"].split(",")))
                for key, value in {
                    "locid": "location_id",
                    "sex__tmp": "sex_id",
                    "age__tmp": "age_group_id",
                }.items():
                    if key in columns:
                        columns.remove(key)
                        columns.add(value)
                if np.any([col in columns for col in ["super_region_id", "region_id"]]):
                    columns.update(["super_region_id", "region_id"])
            else:
                columns.add("location_id")  # swimr default cascade_levels is locid
        elif smoother == "weave":
            for dimension_settings in model_settings["dimensions"].values():
                for key in ["name", "coordinates"]:
                    key_val = dimension_settings.get(key)
                    if key_val:
                        columns.update(as_list(key_val))
    return columns


def get_ensemble_input(config: OneModCFG) -> pd.DataFrame:
    """Get input data for ensemble model."""
    input_cols = set(
        config.col_id + config.col_holdout + [config.col_obs] + config.ensemble.groupby
    )
    rover_input = get_rover_covsel_input(config)
    return rover_input[list(input_cols)]


def get_rover_covsel_submodels(
    experiment_dir: str, save_file: bool = False
) -> list[str]:
    """Get rover submodel IDs and save subsets.
    TODO: merge this to the rover_covsel function to avoid confusion
    """
    dataif, config = get_handle(experiment_dir)

    # Create rover subsets and submodels
    df_input = get_rover_covsel_input(config)
    subsets = Subsets("rover_covsel", config, df_input)
    submodels = [f"subset{subset_id}" for subset_id in subsets.get_subset_ids()]

    # Save file
    if save_file:
        dataif.dump_rover_covsel(subsets.subsets, "subsets.csv")
    return submodels


def get_swimr_submodels(
    experiment_dir: str, save_files: Optional[bool] = False
) -> list[str]:
    """Get swimr submodel IDs; save parameters and subsets."""
    dataif, config = get_handle(experiment_dir)

    # Create swimr parameters, subsets, and submodels
    param_list, subset_list, submodels = [], [], []

    df_input = get_smoother_input("swimr", dataif=dataif, config=config)
    for model_id, model_settings in config["swimr"]["models"].items():
        params = SwimrParams(model_id, model_settings)
        param_list.append(params.param_sets)
        model_settings.inherit()
        subsets = Subsets(model_id, config, df_input)
        subset_list.append(subsets.subsets)
        for param_id, subset_id, holdout_id in product(
            params.get_param_ids(),
            subsets.get_subset_ids(),
            as_list(config.col_holdout) + ["full"],
        ):
            submodel = f"{model_id}_param{param_id}_subset{subset_id}_{holdout_id}"
            submodels.append(submodel)

    # Save files
    if save_files:
        dataif.dump_swimr(pd.concat(param_list), "parameters.csv")
        dataif.dump_swimr(pd.concat(subset_list), "subsets.csv")
    return submodels


def get_weave_submodels(
    experiment_dir: str, save_files: Optional[bool] = False
) -> list[str]:
    """Get weave submodel IDs; save parameters and subsets."""
    dataif, config = get_handle(experiment_dir)

    # Create weave parameters, subsets, and submodels
    param_list, subset_list, submodels = [], [], []

    df_input = get_smoother_input("weave", dataif=dataif, config=config)
    for model_id, model_settings in config["weave"]["models"].items():
        params = WeaveParams(model_id, config)
        param_list.append(params.param_sets)
        model_settings.inherit()
        subsets = Subsets(model_id, model_settings, df_input)
        subset_list.append(subsets.subsets)
        for param_id, subset_id, holdout_id in product(
            params.get_param_ids(),
            subsets.get_subset_ids(),
            as_list(config.col_holdout) + ["full"],
        ):
            for batch_id in subsets.get_batch_ids(subset_id):
                submodel = (
                    f"{model_id}_param{param_id}_subset{subset_id}"
                    f"_{holdout_id}_batch{batch_id}"
                )
                submodels.append(submodel)

    # Save files
    if save_files:
        dataif.dump_weave(pd.concat(param_list), "parameters.csv")
        dataif.dump_weave(pd.concat(subset_list), "subsets.csv")
    return submodels


def get_prediction(row: pd.Series, col_pred: str, model_type: str) -> float:
    """Get smoother prediction."""
    if model_type == "binomial":
        update = row["residual"] * row[col_pred] * (1 - row[col_pred])
        return row[col_pred] + update
    if model_type in ("poisson", "tobit"):
        return (row["residual"] + 1) * row[col_pred]
    raise ValueError("Unsupported model_type")


@cache
def get_handle(experiment_dir: str) -> tuple[DataInterface, OneModCFG]:
    """Get data interface for loading and dumping files. This object encoded the
    folder structure of the experiments, including where the configuration files
    data and results are stored.

    Example
    -------
    >>> experiment_dir = "/path/to/experiment"
    >>> dataif, config = get_handle(experiment_dir)
    >>> df = dataif.load_data()
    >>> df_results = ...
    >>> dataif.dump_rover_covsel(df_results, "results.parquet")

    """
    dataif = DataInterface(experiment=experiment_dir)
    dataif.add_dir("config", dataif.experiment / "config")
    dataif.add_dir("results", dataif.experiment / "results")

    # add settings
    settings_path = dataif.config / "settings.yml"
    if not settings_path.exists():
        raise FileNotFoundError(
            f"please provide a settings file in {str(settings_path)}"
        )
    dataif.add_dir("settings", settings_path)

    # add resources
    resources_path = dataif.config / "resources.yml"
    if not resources_path.exists():
        raise FileNotFoundError(
            f"please provide a resources file in {str(resources_path)}"
        )
    dataif.add_dir("resources", resources_path)

    # add data
    # Initialization stage will ETL input data into a relative path
    data_path = dataif.experiment / "data" / "data.parquet"
    dataif.add_dir("data", data_path)

    # add results folders
    dataif.add_dir("rover_covsel", dataif.results / "rover_covsel")
    dataif.add_dir("regmod_smooth", dataif.results / "regmod_smooth")
    dataif.add_dir("weave", dataif.results / "weave")
    dataif.add_dir("swimr", dataif.results / "swimr")
    dataif.add_dir("ensemble", dataif.results / "ensemble")

    # create confiuration file
    config = OneModCFG(**dataif.load_settings())
    return dataif, config
