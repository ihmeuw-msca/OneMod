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
TODO: Implement priors input

"""

import numpy as np
import pandas as pd
from loguru import logger
from spxmod.model import XModel
from xspline import XSpline

from onemod.config import SpxmodConfig
from onemod.stage import Stage
from onemod.utils.residual import ResidualCalculator


class SpxmodStage(Stage):
    """Spxmod stage."""

    config: SpxmodConfig
    _required_input: list[str] = ["data.parquet"]
    _optional_input: list[str] = [
        "selected_covs.csv",
        "offset.parquet",
        "priors.pkl",
    ]
    _output: list[str] = ["predictions.parquet"]
    _collect_after: list[str] = ["run", "predict"]

    def model_post_init(self, *args, **kwargs) -> None:
        if self.groupby is None:
            raise AttributeError("SPxModStage requires groupby attribute")
        if self.crossby is not None:
            raise AttributeError("SPxModStage does not use crossby attribute")

    def run(self, subset: dict[str, int], *args, **kwargs) -> None:
        """Run spxmod submodel.

        Output
        ------
        model.pkl
            SpXMod model instance for diagnostics.
        coef.csv
            Model coefficients.
        predictions.parquet
            Predictions with residual information.

        """
        # Load data and filter by subset
        data, spline_vars, offset = self._get_submodel_data(subset)

        # Create and fit submodel
        model = self._fit(subset, data, spline_vars, offset)

        # Create submodel predictions
        self._predict(subset, data, model)

    def fit(self, subset: dict[str, int], *args, **kwargs) -> None:
        """Fit spxmod submodel.

        Outputs
        -------
        model.pkl
            SpXMod model instance for diagnostics.
        coef.csv
            Model coefficients.

        """
        # Load data and filter by subset
        logger.info(f"Loading {self.name} data subset {subset}")
        data, spline_vars, offset = self._get_submodel_data(subset)

        # Create and fit submodel
        _ = self._fit(subset, data, spline_vars, offset)

    def _fit(
        self,
        subset: dict[str, int],
        data: pd.DataFrame,
        spline_vars: list[str],
        offset: bool,
    ) -> XModel:
        """Fit spxmod submodel."""
        # Get covariates from upstream stage
        selected_covs = []
        if "selected_covs" in self.input:
            selected_covs = self._get_covs(subset)
            logger.info(
                f"Using covariates from "
                f"{self.input['selected_covs'].stage}: {selected_covs}"
            )

        # Create model parameters
        xmodel_args = self._build_xmodel_args(
            spline_vars, selected_covs, offset
        )
        logger.info(
            f"{len(xmodel_args['var_builders'])} variables created for {self.name}"
        )

        # Create and fit submodel
        logger.info(f"Fitting {self.name} subset {subset}")
        train = data.query(f"{self.config['observation_column']}.notnull()")
        if (train_column := self.config.get("train_column")) is not None:
            train = train.query(f"{train_column} == 1")
        model = XModel.from_config(xmodel_args)
        model.fit(data=train, data_span=data, **self.config.xmodel_fit)

        # Save submodel and coefficients
        logger.info(f"Saving {self.name} subset {subset}")
        self.dataif.dump(
            model, self._get_submodel_dir(subset) + "model.pkl", key="output"
        )
        self.dataif.dump(
            self._get_coef(model),
            self._get_submodel_dir(subset) + "coef.csv",
            key="output",
        )

        return model

    def predict(self, subset: dict[str, int], *args, **kwargs) -> None:
        """Create spxmod submodel predictions.

        Outputs
        -------
        predictions.parquet
            Predictions with residual information.

        """
        # Load data and filter by subset
        data, _, _ = self._get_submodel_data(subset)

        # Load submodel
        logger.info(f"Loading {self.name} subuset {subset}")
        model = self.dataif.load(
            self._get_submodel_dir(subset) + "model.pkl", key="output"
        )

        # Create submodel predictions
        self._predict(subset, data, model)

    def _predict(
        self, subset: dict[str, int], data: pd.DataFrame, model: XModel
    ) -> None:
        """Create spxmod submodel predictions."""
        # Create prediction, residuals, and coefs
        logger.info(f"Creating predictions for {self.name} subset {subset}")
        residual_calculator = ResidualCalculator(self.config["model_type"])
        data[self.config["prediction_column"]] = model.predict(data)
        residuals = residual_calculator.get_residual(
            data,
            self.config["prediction_column"],
            self.config["observation_column"],
            self.config["weights_column"],
        )

        # Save results
        logger.info(f"Saving predictions for {self.name} subset {subset}")
        self.dataif.dump(
            pd.concat([data, residuals], axis=1)[
                list(self.config["id_columns"])
                + ["residual", "residual_se", self.config["prediction_column"]]
            ],
            self._get_submodel_dir(subset) + "predictions.parquet",
            key="output",
        )

    def collect(self) -> None:
        """Collect spxmod submodel results.

        Outputs
        -------
        predictions.parquet
            Predictions with residual information.

        """
        logger.info(f"Collecting {self.name} submodel results")

        # Collect submodel predictions
        if self.subsets is None:
            raise AttributeError("SPxModStage requires subsets")
        self.dataif.dump(
            pd.concat(
                [
                    self.dataif.load(
                        self._get_submodel_dir(subset) + "predictions.parquet",
                        key="output",
                    )
                    for subset in self.subsets.to_dict(orient="records")
                ]
            ),
            "predictions.parquet",
            key="output",
        )

        # TODO: Plot coefficients

    def _get_submodel_data(
        self, subset: dict[str, int]
    ) -> tuple[pd.DataFrame, list[str], bool]:
        """Load submodel data."""
        # Load data and filter by subset
        logger.info(f"Loading {self.name} data subset {subset}")
        data = self.get_subset(self.dataif.load(key="data"), subset)

        # Add spline basis to data
        spline_vars = []
        if self.config.xmodel.spline_config is not None:
            spline_config = self.config.xmodel.spline_config.model_dump()
            col_name = spline_config.pop("name")
            logger.info(f"Getting spline basis for {col_name}")
            spline_basis = self._get_spline_basis(data[col_name], spline_config)
            spline_vars = spline_basis.columns.tolist()
            data = pd.concat([data, spline_basis], axis=1)

        # Add offset to data
        offset = False
        if "offset" in self.input:
            logger.info(f"Adding offset from {self.input['offset'].stage}")
            data = data.merge(
                right=self.dataif.load(
                    columns=list(self.config["id_columns"])
                    + [self.config["prediction_column"]],
                    key="offset",
                    return_type="pandas_dataframe",
                ).rename(columns={self.config["prediction_column"]: "offset"}),
                on=list(self.config["id_columns"]),
                how="left",
            )
            offset = True

        return data, spline_vars, offset

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

    def _get_covs(self, subset: dict[str, int]) -> list[str]:
        """Get covariates from upstream stage."""
        # Load covariates and filter by subset
        selected_covs = self.dataif.load(key="selected_covs")
        cov_groupby = self.get_field(
            field="config:cov_groupby",
            stage_name=self.input["selected_covs"].stage,
        )
        if len(cov_groupby) > 0:
            selected_covs = self.get_subset(
                selected_covs, {key: subset[key] for key in cov_groupby}
            )
        selected_covs = selected_covs["cov"].to_list()

        # Get fixed covariates
        fixed_covs = self.get_field(
            field="config:cov_fixed",
            stage_name=self.input["selected_covs"].stage,
        )
        if "intercept" in fixed_covs:
            fixed_covs.remove("intercept")
        return selected_covs + fixed_covs

    def _build_xmodel_args(
        self, spline_vars: list[str], selected_covs: list[str], offset: bool
    ) -> dict:
        """Format config data for spxmod xmodel.

        Model automatically includes a coefficient for each of the
        selected covariates and age group (based on the 'age_mid' column
        in the input data). Users can also specify intercepts and spline
        variables that vary by dimensions such as age and/or location.

        """
        # Add global settings
        xmodel_args = self.config.xmodel.model_dump(exclude={"spline_config"})
        xmodel_args["model_type"] = self.config["model_type"]
        xmodel_args["obs"] = self.config["observation_column"]
        xmodel_args["weights"] = self.config["weights_column"]

        # Add spline variables, selected_covs, and offset
        if spline_vars:
            xmodel_args = self._add_spline_variables(xmodel_args, spline_vars)
        if selected_covs:
            xmodel_args = self._add_selected_covs(xmodel_args, selected_covs)
        if offset:
            xmodel_args["param_specs"] = {"offset": "offset"}

        # Add coef_bounds and lam to all variables
        coef_bounds = self.config.get("coef_bounds", {})
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
        coefs = []
        for variable in model.var_builders:
            df_sub = variable.space.span.copy()
            df_sub["cov"] = variable.name
            coefs.append(df_sub)
        coef_df = pd.concat(coefs, axis=0, ignore_index=True)
        coef_df["coef"] = model.core.opt_coefs
        return coef_df

    @staticmethod
    def _get_submodel_dir(subset: dict[str, int]) -> str:
        return (
            "submodels/"
            + "_".join(str(value) for value in subset.values())
            + "/"
        )
