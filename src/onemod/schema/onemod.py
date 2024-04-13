from typing import Any, Literal

from onemod.schema.base import Config
from onemod.schema.stages import (
    EnsembleConfig,
    RegmodSmoothConfig,
    RoverCovselConfig,
    WeaveConfig,
)


class OneModConfig(Config):
    """OneMod configuration class. It holds global information about the run. And
    each stage configuration is stored in a separate class.

    Parameters
    ----------
    input_path
        Path to the input data.
    ids
        List of index columns.
    obs
        Observation column name.
    mtype
        Model type. Current options are `'binoimal'`, `'gaussian'`, `'poisson'`.
    weights
        Weights column name.
    pred
        Prediction column name.
    holdouts
        List of holdout columns.
    test
        Test column name. All the observations in the non-test rows must be
        available.
    id_subsets
        Dictionary of subsets with id name as key and list of id values as
        values. This can be used to subset the data.
    rover_covsel
        Rover Covsel stage configuration.
    regmod_smooth
        Regmod Smooth stage configuration.
    weave
        Weave stage configuration.
    ensemble
        Ensemble stage configuration.

    Example
    -------
    This is a sample configuration OneMod model run.

    .. code-block:: yaml

        # Global config
        input_path: /path/to/input/data.parquet
        ids: [age_group_id, location_id, sex_id, year_id]
        id_subsets:
          age_group_id: [8, 9, 10]
          location_id: [13, 14]
          sex_id: [1, 2]
          year_id: [1980, 1981, 1982]
        obs: obs_rate
        mtype: binomial
        weights: sample_size
        pred: pred_rate
        holdouts: [holdout1, holdout2, holdout3]
        test: test

        # Rover covsel settings
        rover_covsel:
          groupby: [age_group_id, sex_id]
          rover:
            cov_fixed: [intercept]
            cov_exploring: [cov1, cov2, cov3]
          rover_fit:
            strategies: [forward]
            top_pct_score: 1.0
            top_pct_learner: 0.5

        # Regmod smooth settings
        regmod_smooth:
          xmodel:
            var_groups:
              - col: "intercept"
              - col: "intercept"
                dim: "super_region_id"
                gprior: [0, 0.35]
            coef_bounds:
              LDI_pc: [-inf, 0]
              education_yrs_pc: [-inf, 0]
            dims:
              - name: "age_mid"
                type: "numerical"
              - name: "super_region_id"
                type: "categorical"
          xmodel_fit:
            options:
              verbose: false
              m_scale: 0.1

        # WeAve settings
        weave:
          model1:
            max_batch: 5000
            groupby: [sex_id, super_region_id]
            dimensions:
              age:
                name: age_group_id
                coordinates: age_mid
                kernel: exponential
                radius: [0.75, 1, 1.25]
              year:
                name: year_id
                kernel: tricubic
                exponent: [0.5, 1, 1.5]
              location:
                name: location_id
                coordinates: [super_region_id, region_id, location_id]
                kernel: depth
                radius: [0.7, 0.8, 0.9]
            model2:
            groupby: [age_group_id, sex_id]
            dimensions:
              year:
                name: year_id
                kernel: tricubic
                exponent: [0.5, 1, 1.5]
              location:
                name: location_id
                kernel: identity
                distance: dictionary
                distance_dict: [/path/to/distance_dict1.parquet, /path/to/distance_dict2.parquet]

        # Ensemble settings
        ensemble:
          groupby: [sex_id, super_region_id]
          metric: rmse
          score: neg_exp
          top_pct_score: 1
          top_pct_model: 1

    """

    input_path: str
    ids: list[str]
    obs: str
    mtype: Literal["binomial", "gaussian", "poisson"]
    weights: str
    pred: str
    holdouts: list[str]
    test: str
    id_subsets: dict[str, list[Any]] = {}

    rover_covsel: RoverCovselConfig | None = None
    regmod_smooth: RegmodSmoothConfig | None = None
    weave: dict[str, WeaveConfig] | None = None
    ensemble: EnsembleConfig | None = None
