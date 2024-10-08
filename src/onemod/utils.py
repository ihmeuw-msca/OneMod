"""Useful functions."""

from __future__ import annotations

import warnings
from functools import cache
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pplkit.data.interface import DataInterface

from onemod.schema import OneModConfig, StageConfig
from onemod.schema.stages import WeaveDimension


class Parameters:
    """Helper class for creating smoother parameter sets.

    Some smoother model parameters are specified as a list. This class
    generates the cartesian product of all possible parameter
    combinations and assigns each a unique parameter ID.

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
        config: OneModConfig | None = None,
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
                raise TypeError(
                    "Settings cannot be None if param_sets is None."
                )
            self.param_sets = self._create_param_sets(config)
        else:
            self.param_sets = param_sets[param_sets["model_id"] == model_id]

    def _create_param_sets(self, config: OneModConfig) -> pd.DataFrame:
        """Create parameter set data frame.

        Parameter set data frame contains parameter IDs and their
        corresponding parameters. Model parameters are not included in
        the data frame if they do not vary by submodel.

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

    def get_param(self, param: str, param_id: int | str) -> Any:
        """Get submodel parameter."""
        params = self.param_sets[self.param_sets["param_id"] == param_id]
        return params[param].item()


class WeaveParams(Parameters):
    """Helper class for creating weave parameter sets."""

    params: tuple[str, ...] = ("radius", "exponent", "distance_dict")

    def _create_param_sets(self, config: OneModConfig) -> pd.DataFrame:
        """Create parameter set data frame."""
        dimensions = config.weave.models[self.model_id].dimensions
        index = pd.MultiIndex.from_product(
            iterables=[
                dimension[param]
                for dimension in dimensions.values()
                for param in self.get_dimension_params(dimension)
            ],
            names=[
                f"{dimension_name}__{param}"
                for dimension_name, dimension in dimensions.items()
                for param in self.get_dimension_params(dimension)
            ],
        )
        param_sets = pd.DataFrame(index=index).reset_index()
        param_cols = list(param_sets.columns)
        param_sets["model_id"] = self.model_id
        param_sets["param_id"] = param_sets.index
        return param_sets[["model_id", "param_id"] + param_cols]

    @classmethod
    def get_dimension_params(cls, dimension: WeaveDimension) -> list[str]:
        """Get dimension parameters by kernel."""
        param_list = []
        if dimension.kernel in ["exponential", "depth", "inverse"]:
            param_list.append("radius")
        elif dimension.kernel == "tricubic":
            param_list.append("exponent")
        if dimension.distance == "dictionary":
            param_list.append("distance_dict")
        return param_list


class Subsets:
    """Helper class for stage subsets based on groupby setting.

    Attributes
    ----------
    stage_id : str
        Stage ID.
    groupby : list of str
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list, which means all points are run in a single model.
    subsets : pandas.DataFrame
        Subset data frame.

    """

    def __init__(
        self,
        stage_id: str,
        stage_config: StageConfig,
        data: pd.DataFrame | None = None,
        subsets: pd.DataFrame | None = None,
    ) -> None:
        """Create Subsets object.

        Parameters
        ----------
        stage_id : str
            Stage ID.
        stage_config : StageConfig
            Stage configuration.
        data : pandas.DataFrame, optional
            Input data.
        subsets : pandas.DataFrame, optional
            Subset data frame.

        """

        self.stage_id = stage_id
        self.groupby = stage_config.groupby
        if subsets is None:
            if data is None:
                raise TypeError(
                    "Data cannot be None if subsets are not provided."
                )
            self.subsets = self._create_subsets(data)
        else:
            self.subsets = subsets[subsets["stage_id"] == stage_id]

    def _create_subsets(self, data: pd.DataFram) -> pd.DataFrame:
        """Create subset data frame.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data.

        Returns
        -------
        pandas.DataFrame
            Subset data frame.

        """
        subset_dict: dict[str, list] = {
            column_name: [] for column_name in self.groupby
        }

        if len(self.groupby) == 0:
            subset_df: pd.DataFrame = pd.DataFrame(
                {"stage_id": [self.stage_id], "subset_id": [0]}
            )
        else:
            for column_vals, _ in data.groupby(self.groupby):
                for idx, column_name in enumerate(self.groupby):
                    subset_dict[column_name].append(column_vals[idx])
            subset_df: pd.DataFrame = pd.DataFrame(subset_dict)
            subset_df["stage_id"] = self.stage_id
            subset_df["subset_id"] = subset_df.index
        return subset_df[["stage_id", "subset_id"] + self.groupby]

    def get_subset_ids(self) -> list:
        """Get list of subset IDs."""
        return self.subsets["subset_id"].tolist()

    def get_column(self, column: str, subset_id: int) -> Any:
        """Get subset column value."""
        subset = self.subsets[self.subsets["subset_id"] == subset_id]
        return subset[column].item()

    def filter_subset(self, data: pd.DataFrame, subset_id: int) -> pd.DataFrame:
        """Filter data by subset."""
        for column in self.groupby:
            data = data[data[column] == self.get_column(column, subset_id)]
        return data


class WeaveSubsets(Subsets):
    """Helper class for weave model subsets based on groupby setting.

    Attributes
    ----------
    stage_id : str
        Stage ID.
    groupby : list of str
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list, which means all points are run in a single model.
    subsets : pandas.DataFrame
        Subset data frame.

    """

    def __init__(
        self,
        stage_id: str,
        stage_config: StageConfig,
        data: pd.DataFrame | None = None,
        subsets: pd.DataFrame | None = None,
    ) -> None:
        """Create Subsets object.

        Parameters
        ----------
        stage_id : str
            Stage ID.
        stage_config : StageConfig
            Stage configuration.
        data : pandas.DataFrame, optional
            Input data.
        subsets : pandas.DataFrame, optional
            Subset data frame.

        """

        super().__init__(stage_id, stage_config, data, subsets)
        if subsets is None:
            n_batches = []
            for subset_id in self.get_subset_ids():
                df = super().filter_subset(data, subset_id)
                if stage_config.max_batch is None:
                    n_batches.append(1)
                else:
                    n_batches.append(
                        int(np.ceil(len(df) / stage_config.max_batch))
                    )
            self.subsets["n_batch"] = n_batches

    def get_batch_ids(self, subset_id: int) -> list:
        """Get list of batch IDs."""
        n_batch = self.subsets[self.subsets["subset_id"] == subset_id][
            "n_batch"
        ].item()
        return np.arange(n_batch).tolist()

    def filter_subset(
        self, data: pd.DataFrame, subset_id: int, batch_id: int = 0
    ) -> pd.DataFrame:
        """Filter data by subset and batch."""
        # TODO: Use itertools.batched in Python 3.12
        data = super().filter_subset(data, subset_id)
        n_batch = self.get_column("n_batch", subset_id)
        max_batch = int(np.ceil(len(data) / n_batch))
        batch_start = batch_id * max_batch
        batch_stop = (batch_id + 1) * max_batch
        data["batch"] = False
        data.iloc[batch_start : min(batch_stop, len(data)), -1] = True
        return data


def as_list(values: Any | list) -> list:
    """Cast values as list if not already."""
    if isinstance(values, (list, tuple, set, dict, np.ndarray)):
        return list(values)
    return [values]


def add_holdouts(
    df: pd.DataFrame,
    n_holdout: int,
    p_holdout: float,
    column: str | None = None,
    seed: int | None = None,
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
            warnings.warn(
                f"Percent held out {p_heldout:.2f} less than {p_holdout}"
            )
        df = df.merge(
            right=pd.concat(df_list)[
                ["age_group_id", "location_id", "sex_id", "year_id", holdout]
            ],
            on=["age_group_id", "location_id", "sex_id", "year_id"],
            how="left",
        )
    return df


def get_weave_input(
    config: OneModConfig,
    dataif: DataInterface,
) -> pd.DataFrame:
    """Get input data for smoother model.

    Each stage only saves ID columns and stage results. Weave needs
    additional columns (e.g., super_region_id, age_mid, holdouts, test)
    that aren't included in spxmod results.

    TODO: Could make a generic version of this function for loading
    predictions from any stage with additional columns from input.

    """
    data = dataif.load_spxmod("predictions.parquet").rename(
        columns={"residual": "spxmod_value", "residual_se": "spxmod_se"}
    )
    columns = _get_weave_columns(data.columns, config)
    data = data.merge(
        right=dataif.load_data()[columns].drop_duplicates(),
        on=config.ids,
    )

    return data


def _get_weave_columns(
    regmod_columns: list[str], config: OneModConfig
) -> list[str]:
    """Get columns needed for weave model."""
    columns = set(config.holdouts + [config.test])
    for model_config in config.weave.models.values():
        columns.update(model_config.groupby)
        for dimension_config in model_config.dimensions.values():
            for key in ["name", "coordinates"]:
                value = dimension_config[key]
                if value:
                    columns.update(as_list(value))
    columns = columns.difference(regmod_columns)
    columns.update(config.ids)
    return list(columns)


def get_submodels(
    stage: str, directory: str, save_file: bool = False
) -> list[str]:
    """Get stage submodel IDs and save subsets."""
    dataif, config = get_handle(directory)
    data = dataif.load_data()
    stage_config = config[stage]

    # Create subsets
    if stage == "weave":
        param_list, subset_list, submodels = [], [], []
        for model_id, model_config in stage_config.models.items():
            params = WeaveParams(model_id, config)
            param_list.append(params.param_sets)
            subsets = WeaveSubsets(model_id, model_config, data)
            subset_list.append(subsets.subsets)
            for param_id, subset_id, holdout_id in product(
                params.get_param_ids(),
                subsets.get_subset_ids(),
                config.holdouts + ["full"],
            ):
                for batch_id in subsets.get_batch_ids(subset_id):
                    submodel = (
                        f"{model_id}__param{param_id}__subset{subset_id}"
                        f"__{holdout_id}__batch{batch_id}"
                    )
                    submodels.append(submodel)
    else:
        subsets = Subsets(stage, stage_config, data)
        submodels = [
            f"subset{subset_id}" for subset_id in subsets.get_subset_ids()
        ]

    # Save file
    if save_file:
        if stage == "weave":
            dataif.dump_weave(pd.concat(param_list), "parameters.csv")
            dataif.dump_weave(pd.concat(subset_list), "subsets.csv")
        else:
            dataif.dump(subsets.subsets, "subsets.csv", key=stage)
    return submodels


def parse_weave_submodel(submodel_id: str) -> dict[str, str | int]:
    """Get IDs from weave submodel_id."""
    id_dict = {}
    submodel_list = submodel_id.split("__")
    id_dict["model_id"] = submodel_list[0]
    id_dict["param_id"] = int(submodel_list[1].removeprefix("param"))
    id_dict["subset_id"] = int(submodel_list[2].removeprefix("subset"))
    id_dict["holdout_id"] = submodel_list[3]
    id_dict["batch_id"] = int(submodel_list[4].removeprefix("batch"))
    return id_dict


def reformat_weave_results(
    df_pivot: pd.DataFrame, columns: str | list[str], ids: list[str]
) -> pd.DataFrame:
    """Reformat MultiIndex weave results.

    To reduce file sizes, weave results are saved with a MultiIndex for
    indices (config.col_ids) and columns (results, model_id, param_id).
    This function returns a data frame with columns ids + [model_id,
    param_id] + columns.

    Parameters
    ----------
    df_pivot : pandas.DataFrame
        MultiIndex weave results.
    columns : str or list of str
        Name of result columns to return.
    ids : list of str
        Index column names.

    Returns
    -------
    pandas.DataFrame

    """
    columns = as_list(columns)
    df_melt = _reformat_weave_column(df_pivot, columns[0])
    for column in columns[1:]:
        df_melt = df_melt.merge(
            right=_reformat_weave_column(df_pivot, column),
            on=ids + ["model_id", "param_id"],
        )
    return df_melt


def _reformat_weave_column(df: pd.DataFrame, column=str) -> pd.DataFrame:
    """Reformat single column from MultiIndex weave results.

    Parameters
    ----------
    df_pivot : pandas.DataFrame
        MultiIndex weave results.
    columns : str or list of str
        Name of result column to return.

    Returns
    -------
    pandas.DataFrame
        Data frame with index columns and specified result column.

    """
    return df[column].melt(value_name=column, ignore_index=False).reset_index()


@cache
def get_handle(directory: str) -> tuple[DataInterface, OneModConfig]:
    """Get data interface for loading and dumping files. This object encoded the
    folder structure of the experiments, including where the configuration files
    data and results are stored.

    Example
    -------
    >>> directory = "/path/to/experiment"
    >>> dataif, config = get_handle(directory)
    >>> df = dataif.load_data()
    >>> df_results = ...
    >>> dataif.dump_rover_covsel(df_results, "results.parquet")

    """
    dataif = DataInterface(experiment=directory)
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
    dataif.add_dir("spxmod", dataif.results / "spxmod")
    dataif.add_dir("weave", dataif.results / "weave")
    dataif.add_dir("ensemble", dataif.results / "ensemble")

    # create confiuration file
    config = OneModConfig(**dataif.load_settings())

    # For pre-workflow use of data, add raw data path
    dataif.add_dir("raw_data", config.input_path)

    return dataif, config


def format_input(directory: str) -> None:
    """Filter input data by ID subsets and add test column.

    Parameters
    ----------
    directory : str
        The experiment directory.

    """
    # Load input data
    dataif, config = get_handle(directory)
    data = dataif.load(config.input_path)

    # Filter data by ID subsets
    if config.id_subsets:
        data = data.query(
            " & ".join(
                [
                    f"{key}.isin({value})"
                    for key, value in config.id_subsets.items()
                ]
            )
        ).reset_index(drop=True)

    # Create a test column if not in input
    if config.test not in data:
        logger.warning(
            "Test column not found; setting null observations as test rows."
        )
        data[config.test] = data[config.obs].isna().astype("int")

    # TODO: Create holdout columns if not in input

    # Save to directory/data/data.parquet
    dataif.dump_data(data)
