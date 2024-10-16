"""ModRover covariate selection stage.

TODO: Update read/write statements with DataInterface

"""

import warnings
import json
from loguru import logger

import dill
import fire
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

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run rover submodel."""
        self.fit(subset_id)

    def fit(self, subset_id: int | None = None) -> None:
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
        logger.info(f"Loading rover data subset {subset_id}")
        data = self.get_stage_subset(subset_id).query(
            f"{self.config.test_column} == 0"
        )

        # Fit submodel
        if len(data) > 0:
            logger.info(f"Fitting rover submodel {subset_id}")

            # Create rover submodel
            submodel = Rover(
                obs=self.config.observation_column,
                model_type=self.config.model_type,
                cov_fixed=list(self.config.cov_fixed),
                cov_exploring=list(self.config.cov_exploring),
                weights=self.config.weight_column,
                holdouts=list(self.config.holdout_columns),
            )

            # Fit rover submodel
            submodel.fit(
                data=data,
                strategies=list(self.config.strategies),
                top_pct_score=self.config.top_pct_score,
                top_pct_learner=self.config.top_pct_learner,
                coef_bounds=self.config.coef_bounds or {},
            )

            # Save results
            logger.info(f"Saving rover submodel {subset_id} results")
            submodel_dir = self.directory / "submodels" / subset_id
            submodel.learner_info.to_csv(
                submodel_dir / "learner_info.csv", index=False
            )
            with open(submodel_dir / "rover.pkl", "wb") as f:
                dill.dump(submodel, f)
            submodel.summary.to_csv(submodel_dir / "summary.csv", index=False)
        else:
            logger.info(f"No training data for rover submodel {subset_id}")

    def collect(self) -> None:
        """Collect rover submodel results."""
        # Concatenate summaries
        logger.info("Concatenating rover coefficient summaries")
        summaries = self._get_rover_summaries()
        summaries.to_csv(self.directory / "summaries.csv", index=False)

        # Select covariates
        logger.info("Selecting rover covariates")
        selected_covs = self._get_selected_covs(summaries)
        selected_covs.to_csv(self.directory / "selected_covs", index=False)

        # Plot covariates
        # TODO: Implement plotting functions
        logger.info("Plotting rover covariates")

    def _get_rover_summaries(self) -> pd.DataFrame:
        """Concatenate rover coefficient summaries.

        What if rover doesn't have groupby?

        """
        subsets = pd.read_csv(self.directory / "subsets.csv")

        # Collect coefficient summaries
        summaries = []
        for subset_id in self.subset_ids:
            try:
                summary = pd.read_csv(
                    self.directory / "submodels" / subset_id / "summary.csv"
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
        """Select rover covariates.

        TODO: Document how things work with pipeline

        What if pipeline doesn't have groupby?

        """
        if self.pipeline is None:
            raise NotImplementedError()
        pipeline_groupby = self._get_pipeline_groupby()
        if pipeline_groupby is None:
            raise NotImplementedError()
        selected_covs = []
        for group, df in summaries.groupby(pipeline_groupby):
            t_stats = (
                df.groupby(pipeline_groupby + ["cov"])["abs_t_stat"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
                .eval(f"selected = abs_t_stat >= {self.config.t_threshold}")
            )
            if (
                self.config.min_covs is not None
                and t_stats["selected"].sum() < self.config.min_covs
            ):
                t_stats.loc[: self.config.min_covs - 1, "selected"] = True
            if (
                self.config.max_covs is not None
                and t_stats["selected"].sum > self.config.max_covs
            ):
                t_stats.loc[self.config.max_covs :, "selected"] = True
            selected = t_stats.query("selected").drop(columns="selected")
            selected_covs.append(selected)
            logger.info(
                ", ".join(
                    [
                        f"{id_name}: {id_val}"
                        for id_name, id_val in zip(pipeline_groupby, group)
                    ]
                    + [f"covs: {selected['cov'].values}"]
                )
            )
        return pd.concat(selected_covs)

    def _get_pipeline_groupby(self) -> set[str] | None:
        """Get pipeline groupby attribute.

        TODO: Any error messages?

        """
        with open(self.directory.parent / (self.pipeline + ".json"), "r") as f:
            config = json.load(f)
        return config.get("groupby")


if __name__ == "__main__":
    fire.Fire(RoverStage.evaluate)
