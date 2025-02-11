"""ModRover covariate selection stage.

Notes
-----
* RoverStage currently requires nonempty `groupby` attribute.
* Covariates are selected for subsets based on `cov_groupby`. For
  example, if `stage.config['cov_groupby']` is 'sex_id' and
  `stage.groupby` is `['sex_id', 'age_group_id']`, submodels will be fit
  separately for each age/sex pair, and covariates will be selected
  separately for each sex.

"""

import warnings

import pandas as pd
from loguru import logger
from modrover.api import Rover

from onemod.config import RoverConfig
from onemod.stage import Stage


class RoverStage(Stage):
    """ModRover covariate selection stage."""

    config: RoverConfig
    _skip: list[str] = ["predict"]
    _required_input: list[str] = ["data.parquet"]
    _output: list[str] = ["selected_covs.csv", "summaries.csv"]
    _collect_after: list[str] = ["run", "fit"]

    def model_post_init(self, *args, **kwargs) -> None:
        if self.groupby is None:
            raise AttributeError("RoverStage requires groupby attribute")
        if self.crossby is not None:
            raise AttributeError("RoverStage does not use crossby attribute")

    def _run(self, subset: dict[str, int], *args, **kwargs) -> None:
        """Run rover submodel."""
        self._fit(subset)

    def _fit(self, subset: dict[str, int], *args, **kwargs) -> None:
        """Fit rover submodel.

        Outputs
        -------
        learner_info.csv
            Information about every learner.
        model.pkl
            Rover object used for plotting and diagnostics.
        summary.csv
            Summary covariate coefficients from the ensemble model.

        """
        # Load data and filter by subset
        logger.info(f"Loading {self.name} data subset {subset}")
        train = self.dataif.load(key="data", subset=subset).query(
            f"{self.config['observation_column']}.notnull()"
        )
        if (train_column := self.config.get("train_column")) is not None:
            train = train.query(f"{train_column} == 1")

        if len(train) > 0:
            logger.info(f"Fitting {self.name} subset {subset}")

            # Create submodel
            submodel = Rover(
                obs=self.config["observation_column"],
                model_type=self.config["model_type"],
                cov_fixed=list(self.config["cov_fixed"]),
                cov_exploring=list(self.config["cov_exploring"]),
                weights=self.config["weights_column"],
                holdouts=list(self.config["holdout_columns"]),
            )

            # Fit submodel
            submodel.fit(
                data=train,
                strategies=list(self.config.strategies),
                top_pct_score=self.config.top_pct_score,
                top_pct_learner=self.config.top_pct_learner,
                coef_bounds=self.config.get("coef_bounds", {}),
            )

            # Save results
            logger.info(f"Saving {self.name} subset {subset} results")
            self.dataif.dump(
                submodel.learner_info,
                self._get_submodel_dir(subset) + "learner_info.csv",
                key="output",
            )
            self.dataif.dump(
                submodel,
                self._get_submodel_dir(subset) + "model.pkl",
                key="output",
            )
            self.dataif.dump(
                submodel.summary,
                self._get_submodel_dir(subset) + "summary.csv",
                key="output",
            )
        else:
            logger.info(f"No training data for {self.name} subset {subset}")

    def collect(self) -> None:
        """Collect rover submodel results.

        Outputs
        -------
        selected_covs.csv
            Covariates selected for subsets based on pipeline.groupby.
        summaries.csv
            Covariate coefficient summaries from the submodel ensemble
            models.

        """
        # Concatenate summaries
        logger.info(f"Concatenating {self.name} coefficient summaries")
        summaries = self._get_rover_summaries()
        self.dataif.dump(summaries, "summaries.csv", key="output")

        # Select covariates
        logger.info(f"Selecting {self.name} covariates")
        selected_covs = self._get_selected_covs(summaries)
        self.dataif.dump(selected_covs, "selected_covs.csv", key="output")

        # TODO: Plot covariates

    def _get_rover_summaries(self) -> pd.DataFrame:
        """Concatenate rover coefficient summaries."""
        if self.subsets is None:
            raise AttributeError("RoverStage requires subsets")

        # Collect coefficient summaries
        summaries = []
        for subset in self.subsets.to_dict(orient="records"):
            try:
                summary = self.dataif.load(
                    self._get_submodel_dir(subset) + "summary.csv",  # type: ignore
                    key="output",
                )
                for key, value in subset.items():
                    summary[key] = value
                summaries.append(summary)
            except FileNotFoundError:
                warnings.warn(f"Rover subset {subset} missing summary.csv")
        summaries_df = pd.concat(summaries)

        # Add t-statistic
        summaries_df["abs_t_stat"] = (
            summaries_df["coef"].abs() / summaries_df["coef_sd"]
        )
        return summaries_df

    def _get_selected_covs(self, summaries: pd.DataFrame) -> pd.DataFrame:
        """Select rover covariates."""
        selected_covs = []

        if len(cov_groupby := self.config["cov_groupby"]) > 0:
            for subset, subset_summaries in summaries.groupby(cov_groupby):
                subset_selected_covs = self._get_subset_selected_covs(
                    subset_summaries, cov_groupby
                )
                selected_covs.append(subset_selected_covs)

                logger.info(
                    ", ".join(
                        [
                            f"{id_name}: {id_val}"
                            for id_name, id_val in zip(cov_groupby, subset)
                        ]
                        + [f"covs: {subset_selected_covs['cov'].values}"]
                    )
                )
        else:
            selected_covs.append(self._get_subset_selected_covs(summaries, []))
            logger.info(f"covs: {selected_covs[0]['cov'].values}")

        return pd.concat(selected_covs, ignore_index=True)

    def _get_subset_selected_covs(
        self, subset_summaries: pd.DataFrame, groupby: list[str]
    ) -> pd.DataFrame:
        """Select rover covariates for data subset."""
        # Select covariates greater than t_threshold
        t_stats = (
            subset_summaries.groupby(groupby + ["cov"])["abs_t_stat"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        t_stats["selected"] = t_stats["abs_t_stat"] >= self.config.t_threshold

        # Add/remove covariates based on min_covs/max_covs
        if (
            self.config.min_covs is not None
            and t_stats["selected"].sum() < self.config.min_covs
        ):
            t_stats.loc[: self.config.min_covs - 1, "selected"] = True
        if (
            self.config.max_covs is not None
            and t_stats["selected"].sum() > self.config.max_covs
        ):
            t_stats.loc[self.config.max_covs :, "selected"] = False

        return t_stats.query("selected").drop(columns="selected")

    @staticmethod
    def _get_submodel_dir(subset: dict[str, int]) -> str:
        return (
            "submodels/"
            + "_".join(str(value) for value in subset.values())
            + "/"
        )
