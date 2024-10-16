"""ModRover covariate selection stage."""

from loguru import logger

import dill
import fire
from modrover.api import Rover

from onemod.config import RoverConfig
from onemod.stage import ModelStage


class RoverStage(ModelStage):
    """ModRover covariate selection stage."""

    config: RoverConfig
    _skip: set[str] = {"predict"}
    _required_input: set[str] = {"data.parquet"}
    _output: set[str] = {
        "learner_info.csv",
        "rover.pkl",
        "selected_covs.csv",
        "summary.csv",
    }

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Run rover submodel."""
        self.fit(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        """Fit rover submodel.

        Outputs
        -------
        learner_info.csv
            Information about every learner.
        rover.pkl
            Rover object used for plotting and diagnostics.
        selected_covs.csv
            Selected covariates.
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
                cov_fixed=self.config.cov_fixed,
                cov_exploring=self.config.cov_exploring,
                weights=self.config.weights,
                holdouts=self.config.holdout_columns,
            )

            # Fit rover submodel
            submodel.fit(
                data=data,
                strategies=self.config.strategies,
                top_pct_score=self.config.top_pct_score,
                top_pct_learner=self.config.top_pct_learner,
                coef_bounds=self.config.coef_bounds or {},
            )

            # Save results
            # TODO: Simplify with DataInterface
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
        print(f"collecting {self.name} submodel results")
