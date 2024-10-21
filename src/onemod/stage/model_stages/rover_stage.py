"""ModRover covariate selection stage.

Notes
-----
* RoverStage currently requires nonempty `groupby` attribute.
* Covariates are selected for subsets based on pipeline.groupby. For
  example, if pipeline.groupby is 'sex_id' and stage.groupby is
  'age_group_id', submodels will be fit separately for each age/sex
  pair, and covariates will be selected separately for each sex.

TODO: Update read/write statements with DataInterface

"""

import warnings
import json
from loguru import logger

import dill
import pandas as pd
from modrover.api import Rover

from onemod.config import RoverConfig
from onemod.stage import ModelStage


class RoverStage(ModelStage):
    """ModRover covariate selection stage."""

    config: RoverConfig
    _skip: set[str] = {"predict"}
    _required_input: set[str] = {"data.parquet"}
    _output: set[str] = {"selected_covs.csv", "summaries.csv"}

    def run(self, subset_id: int, *args, **kwargs) -> None:
        """Run rover submodel."""
        self.fit(subset_id)

    def fit(self, subset_id: int, *args, **kwargs) -> None:
        """Fit rover submodel.

        Outputs
        -------
        learner_info.csv
            Information about every learner.
        rover.pkl
            Rover object used for plotting and diagnostics.
        summary.csv
            Summary covariate coefficients from the ensemble model.

        """
        # Load data and filter by subset
        logger.info(f"Loading {self.name} data subset {subset_id}")
        data = self.get_stage_subset(subset_id).query(
            f"{self.config.test_column} == 0"
        )

        if len(data) > 0:
            logger.info(f"Fitting {self.name} submodel {subset_id}")

            # Create submodel
            submodel = Rover(
                obs=self.config.observation_column,
                model_type=self.config.model_type,
                cov_fixed=list(self.config.cov_fixed),
                cov_exploring=list(self.config.cov_exploring),
                weights=self.config.weight_column,
                holdouts=list(self.config.holdout_columns),
            )

            # Fit submodel
            submodel.fit(
                data=data,
                strategies=list(self.config.strategies),
                top_pct_score=self.config.top_pct_score,
                top_pct_learner=self.config.top_pct_learner,
                coef_bounds=self.config.coef_bounds or {},
            )

            # Save results
            logger.info(f"Saving {self.name} submodel {subset_id} results")
            submodel_dir = self.directory / "submodels" / str(subset_id)
            submodel_dir.mkdir(exist_ok=True)
            submodel.learner_info.to_csv(
                submodel_dir / "learner_info.csv", index=False
            )
            with open(submodel_dir / "rover.pkl", "wb") as f:
                dill.dump(submodel, f)
            submodel.summary.to_csv(submodel_dir / "summary.csv", index=False)
        else:
            logger.info(
                f"No training data for {self.name} submodel {subset_id}"
            )

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
        summaries.to_csv(self.directory / "summaries.csv", index=False)

        # Select covariates
        logger.info(f"Selecting {self.name} covariates")
        selected_covs = self._get_selected_covs(summaries)
        selected_covs.to_csv(self.directory / "selected_covs.csv", index=False)

        # TODO: Plot covariates

    def _get_rover_summaries(self) -> pd.DataFrame:
        """Concatenate rover coefficient summaries."""
        subsets = pd.read_csv(self.directory / "subsets.csv")

        # Collect coefficient summaries
        summaries = []
        for subset_id in self.subset_ids:
            try:
                summary = pd.read_csv(
                    self.directory
                    / "submodels"
                    / str(subset_id)
                    / "summary.csv"
                )
                summary["subset_id"] = subset_id
                summaries.append(summary)
            except FileNotFoundError:
                warnings.warn(f"Rover submodel {subset_id} missing summary.csv")
        summaries = pd.concat(summaries)

        # Merge with subsets and add t-statistic
        summaries = summaries.merge(subsets, on="subset_id", how="left")
        summaries["abs_t_stat"] = summaries.eval("abs(coef / coef_sd)")
        return summaries

    def _get_selected_covs(self, summaries: pd.DataFrame) -> pd.DataFrame:
        """Select rover covariates."""
        pipeline_groupby = self.get_pipeline_groupby()
        if pipeline_groupby:
            selected_covs = []
            for subset, subset_summaries in summaries.groupby(pipeline_groupby):
                subset_selected_covs = self._get_subset_selected_covs(
                    subset_summaries, pipeline_groupby
                )
                selected_covs.append(subset_selected_covs)
                logger.info(
                    ", ".join(
                        [
                            f"{id_name}: {id_val}"
                            for id_name, id_val in zip(pipeline_groupby, subset)
                        ]
                        + [f"covs: {subset_selected_covs['cov'].values}"]
                    )
                )
            return pd.concat(selected_covs)
        selected_covs = self._get_subset_selected_covs(summaries, [])
        logger.info(f"covs: {selected_covs['cov'].values}")
        return selected_covs

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
            .eval(f"selected = abs_t_stat >= {self.config.t_threshold}")
        )

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
