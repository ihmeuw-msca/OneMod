"""Spxmod stage.

This stage fits a model using priors to smooth covariate coefficients
across age groups (based on the 'age_mid' column in the input data).
Users can also specify intercepts and spline variables that vary by
dimensions such as age and/or location.

Notes
-----
* SpXModStage currently requires nonempty `groupby` attribute.

TODO: Update for new spxmod version with splines
TODO: Make selected_covs more flexible, not hard-coded to age_mid
TODO: Implement offset and priors input

"""

import numpy as np
import pandas as pd
from loguru import logger
from spxmod.model import XModel
from xspline import XSpline

from onemod.config import SpxmodConfig
from onemod.stage import ModelStage
from onemod.utils.residual import ResidualCalculator
from onemod.utils.subsets import get_subset


class SpxmodStage(ModelStage):
    """Spxmod stage."""

    config: SpxmodConfig
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {
        "selected_covs.csv",
        "offset.parquet",
        "priors.pkl",
    }
    _output: set[str] = {"predictions.parquet"}
    _collect_after: set[str] = {"run", "predict"}

    def run(self, subset_id: int, *args, **kwargs) -> None:
        """Run spxmod submodel."""
        self.fit(subset_id)
        self.predict(subset_id)

    def fit(self, subset_id: int, *args, **kwargs) -> None:
        """Fit spxmod submodel.

        Outputs
        -------
        model.pkl
            SpXMod model instance for diagnostics.
        coef.csv
            Model coefficients for different age groups.

        """
        # Load data and filter by subset
        logger.info(f"Loading {self.name} data subset {subset_id}")
        data = self.get_stage_subset(subset_id)

        # Add spline basis to data
        spline_vars = []
        if self.config.xmodel.spline_config is not None:
            spline_config = self.config.xmodel.spline_config.model_dump()
            col_name = spline_config.pop("name")
            logger.info(f"Getting spline basis for {col_name}")
            spline_basis = self._get_spline_basis(data[col_name], spline_config)
            spline_vars = spline_basis.columns.tolist()
            data = pd.concat([data, spline_basis], axis=1)

        # Get covariates from upstream stage
        selected_covs = []
        if "selected_covs" in self.input:
            selected_covs = self._get_covs(subset_id)
            logger.info(
                f"Using covariates selected by "
                f"{self.input['selected_covs'].stage}: {selected_covs}"
            )

        # Create model parameters
        xmodel_args = self._build_xmodel_args(selected_covs, spline_vars)
        logger.info(
            f"{len(xmodel_args['var_builders'])} variables created for {self.name}"
        )

        # Create and fit submodel
        logger.info(f"Fitting {self.name} submodel {subset_id}")
        train = data.query(
            f"({self.config["test_column"]} == 0) & {self.config["observation_column"]}.notnull()"
        )
        model = XModel.from_config(xmodel_args)
        model.fit(data=train, data_span=data, **self.config.xmodel_fit)

        # Save submodel
        logger.info(f"Saving {self.name} submodel {subset_id}")
        self.dataif.dump_output(model, f"submodels/{subset_id}/model.pkl")

    def predict(self, subset_id: int, *args, **kwargs) -> None:
        """Create spxmod submodel predictions.

        Outputs
        -------
        predictions.parquet
            Predictions with residual information.

        """
        # TODO: Make function to do this since in both places
        # Load data and filter by subset
        data = self.get_stage_subset(subset_id)

        # Add spline basis to data
        if self.config.xmodel.spline_config is not None:
            spline_config = self.config.xmodel.spline_config.model_dump()
            col_name = spline_config.pop("name")
            logger.info(f"Getting spline basis for {col_name}")
            spline_basis = self._get_spline_basis(data[col_name], spline_config)
            data = pd.concat([data, spline_basis], axis=1)

        # Load submodel
        model = self.dataif.load_output(f"submodels/{subset_id}/model.pkl")

        # Create prediction, residuals, and coefs
        logger.info("Calculating predictions and residuals")
        residual_calculator = ResidualCalculator(self.config["model_type"])
        data[self.config["prediction_column"]] = model.predict(data)
        residuals = residual_calculator.get_residual(
            data,
            self.config["prediction_column"],
            self.config["observation_column"],
            self.config["weights_column"],
        )
        coefs = self._get_coef(model)

        # Save results
        self.dataif.dump_output(
            pd.concat([data, residuals], axis=1)[
                list(self.config["id_columns"])
                + ["residual", "residual_se", self.config["prediction_column"]]
            ],
            f"submodels/{subset_id}/predictions.parquet",
        )
        self.dataif.dump_output(coefs, f"submodels/{subset_id}/coef.csv")

    def collect(self) -> None:
        """Collect spxmod submodel results.

        Outputs
        -------
        predictions.parquet
            Predictions with residual information.

        """
        # Collect submodel predictions
        self.dataif.dump_output(
            pd.concat(
                [
                    self.dataif.load_output(
                        f"submodels/{subset_id}/predictions.parquet"
                    )
                    for subset_id in self.subset_ids
                ]
            ),
            "predictions.parquet",
        )

        # TODO: Plot coefficients

    @staticmethod
    def _get_spline_basis(
        column: pd.Series, spline_config: dict
    ) -> pd.DataFrame:
        """Get spline basis based on data and configuration."""
        col_min, col_max = column.min(), column.max()
        spline_config["knots"] = col_min + np.array(spline_config["knots"]) * (
            col_max - col_min
        )
        spline = XSpline(**spline_config)
        idx_start = 0 if spline_config["include_first_basis"] else 1
        spline_basis = pd.DataFrame(
            spline.design_mat(column),
            index=column.index,
            columns=[
                f"spline_{ii+idx_start}"
                for ii in range(spline.num_spline_bases)
            ],
        )
        return spline_basis

    def _get_covs(self, subset_id: int) -> list[str]:
        """Get covariates from upstream stage."""
        # Load covariates and filter by subset
        selected_covs = self.dataif.load_selected_covs()
        if pipeline_groupby := self.get_field("groupby"):
            selected_covs = get_subset(
                selected_covs,
                self.dataif.load_output("subsets.csv"),
                subset_id,
                id_names=pipeline_groupby,
            )
        selected_covs = selected_covs["cov"].tolist()

        # Get fixed covariates
        fixed_covs = self.get_field(
            field="config-cov_fixed",
            stage_name=self.input["selected_covs"].stage,
        )
        if "intercept" in fixed_covs:
            fixed_covs.remove("intercept")
        return selected_covs + fixed_covs

    def _build_xmodel_args(
        self, selected_covs: list[str], spline_vars: list[str]
    ) -> dict:
        """Format config data for spxmod xmodel.

        Model automatically includes a coefficient for each of the
        selected covariates and age group (based on the 'age_mid' column
        in the input data). Users can also specify intercepts and spline
        variables that vary by dimensions such as age and/or location.

        """
        # Add global settings
        xmodel_args = self.config.xmodel.model_dump(exclude="spline_config")
        xmodel_args["model_type"] = self.config["model_type"]
        xmodel_args["obs"] = self.config["observation_column"]
        xmodel_args["weights"] = self.config["weights_column"]

        # Add covariate and spline variables
        if selected_covs:
            xmodel_args = self._add_selected_covs(xmodel_args, selected_covs)
        if spline_vars:
            xmodel_args = self._add_spline_variables(xmodel_args, spline_vars)

        # Add coef_bounds and lam to all variables
        coef_bounds = self.config["coef_bounds"] or {}
        lam = xmodel_args.pop("lam")
        xmodel_args = self._add_prior_settings(xmodel_args, coef_bounds, lam)

        # Add dummy space for any regular variables
        add_dummy = False
        for var in xmodel_args["variables"]:
            if var["space"] is None:
                var["space"] = "dummy"
                add_dummy = True
        if add_dummy:
            xmodel_args["spaces"].append({"name": "dummy", "dims": []})

        xmodel_args["var_builders"] = xmodel_args.pop("variables")

        return xmodel_args

    @staticmethod
    def _add_selected_covs(xmodel_args: dict, selected_covs: list[str]) -> dict:
        """Add selected covariates to spxmod model configuration."""
        # add age_mid to spaces if not already included
        space_keys = [space["name"] for space in xmodel_args["spaces"]]
        if "age_mid" not in space_keys:
            xmodel_args["spaces"].append(
                dict(
                    name="age_mid",
                    dims=[dict(name="age_mid", type="numerical")],
                )
            )

        # add variables for selected covs if not already included
        variable_keys = [
            (variable["name"], variable["space"])
            for variable in xmodel_args["variables"]
        ]
        for cov in selected_covs:
            if (cov, "age_mid") not in variable_keys:
                xmodel_args["variables"].append(dict(name=cov, space="age_mid"))

        return xmodel_args

    @staticmethod
    def _add_spline_variables(
        xmodel_args: dict, spline_vars: list[str]
    ) -> dict:
        """Add spline variables to spxmod model configuration."""
        for var in xmodel_args["variables"].copy():
            if var["name"] == "spline":
                xmodel_args["variables"].remove(var)
                for spline_var in spline_vars:
                    spline_variable = var.copy()
                    spline_variable["name"] = spline_var
                    xmodel_args["variables"].append(spline_variable)
        return xmodel_args

    @staticmethod
    def _add_prior_settings(
        xmodel_args: dict, coef_bounds: dict, lam: float
    ) -> dict:
        """Add coef_bounds and lam to all variables."""
        for variable in xmodel_args["variables"]:
            cov = variable["name"]
            if "uprior" not in variable or variable["uprior"] is None:
                variable["uprior"] = coef_bounds.get(cov)
            if "lam" not in variable or variable["lam"] is None:
                variable["lam"] = lam
        return xmodel_args

    @staticmethod
    def _get_coef(model: XModel) -> pd.DataFrame:
        """Get coefficient information from the trained model.

        Parameters
        ----------
        model : XModel
            The statistical model object containing coefficient data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing coefficient, dimension, and dimension
            value information.

        """
        df_coef = []
        for variable in model.var_builders:
            df_sub = variable.space.span.copy()
            df_sub["cov"] = variable.name
            df_coef.append(df_sub)
        df_coef = pd.concat(df_coef, axis=0, ignore_index=True)
        df_coef["coef"] = model.core.opt_coefs
        return df_coef
