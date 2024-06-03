from typing import Any, Literal

from onemod.schema.base import Config, StageConfig
from onemod.schema.stages import (
    EnsembleConfig,
    RoverCovselConfig,
    SPxModConfig,
    WeaveConfig,
)


class OneModConfig(Config):
    """OneMod configuration class.

    This class holds the global configuration information for a OneMod
    experiment. Each stage configuration is stored in a separate class.

    Parameters
    ----------
    input_path
        Path to the input data.
    ids
        List of index columns.
    obs
        Observation column name.
    mtype
        Model type. Current options are `'binoimal'`, `'gaussian'`, and
        `'poisson'`.
    weights
        Weights column name.
    pred
        Prediction column name.
    holdouts
        List of holdout columns. In the input data, holdout columns
        should have values 0 (train), 1 (holdout), or NaN (missing
        observations). Holdout sets are used to evaluate out-of-sample
        performance for rover and weave models.
    test
        Test column name. In the input data, the test column should have
        values 0 (train) or 1 (test). Test data is never used to train
        any stage models, so it can be used to evaluate out-of-sample
        performance of the entire onemod pipeline. If no test column is
        provided, all missing observations will be the test set.
    id_subsets
        Dictionary of subsets with ID name as key and list of ID values
        as values. This can be used to run models on subsets of the
        input data; for example, if the input data includes data from
        1950-2000 but you only want to run a model from 1980-1990.
    groupby
        List of ID columns to group data by when running separate models
        for each sex_id, age_group_id, super_region_id, etc. Default is
        an empty list, which means all points are run in a single model.
        This setting applies to all stages.
    rover_covsel
        Rover covariate selection stage configuration.
    spxmod
        SPxMod stage configuration.
    weave
        Weave stage configuration.
    ensemble
        Ensemble stage configuration.

    TODO: Indicate which settings are used by which stages.

    Examples
    --------
    This is a sample OneMod configuration. All stages run separate
    models by sex_id because of the OneMod groupby setting.

    .. code-block:: yaml

        # OneMod settings
        input_path: /path/to/input/data.parquet
        ids: [age_group_id, location_id, sex_id, year_id]
        id_subsets:
          age_group_id: [8, 9, 10]
          location_id: [13, 14]
          sex_id: [1, 2]
          year_id: [1980, 1981, 1982]
        groupby: [sex_id]
        obs: obs_rate
        mtype: binomial
        weights: sample_size
        pred: pred_rate
        holdouts: [holdout1, holdout2, holdout3]
        test: test

        # Rover covariate selection settings
        rover_covsel:
          groupby: [age_group_id]
          t_threshold: 1.0
          rover:
            cov_fixed: [intercept]
            cov_exploring: [cov1, cov2, cov3]
          rover_fit:
            strategies: [forward]
            top_pct_score: 1.0
            top_pct_learner: 0.5

        # SPxMod settings
        spxmod:
          xmodel:
            spaces:
            - name: age_mid
              dims:
                - name: age_mid
                  dim_type: numerical
            - name: super_region_id*agd_mid
              dims:
                - name: super_region_id
                  dim_type: categorical
                - name: age_mid
                  dim_type: numerical
            var_builders:
              - name: intercept
                space: super_region_id*age_mid
                lam: 1.0
                # above lam is equivalent to use gprior as follows
                # gprior:
                #   mean: 0.0
                #   sd: 1.0
            coef_bounds:
              LDI_pc:
                # lower bounds is automatically set to -inf
                ub: 0.0
              smoking_prev:
                # upper bounds is automatically set to inf
                lb: 0.0
            lam: 100.0
          xmodel_fit:
            options:
              verbose: false
              # this is only used if we set bounds in the var_builders, this is
              # the settings for interior point optimization, for more please
              # check `here <https://github.com/ihmeuw-msca/msca/blob/main/src/msca/optim/solver/ipsolver.py#L158>`_.
              m_scale: 0.1

        # WeAve settings
        weave:
          models:
            super_region_model:
              groupby: [super_region_id]
              max_batch: 5000
              dimensions:
                age:
                  name: age_group_id
                  coordinates: [age_mid]
                  kernel: exponential
                  radius: [0.75, 1, 1.25]
                location:
                  name: location_id
                  coordinates: [super_region_id, region_id, location_id]
                  kernel: depth
                  radius: [0.7, 0.8, 0.9]
                year:
                  name: year_id
                  kernel: tricubic
                  exponent: [0.5, 1, 1.5]
              age_group_model:
                groupby: [age_group_id]
                dimensions:
                  location:
                    name: location_id
                    kernel: identity
                    distance: dictionary
                    distance_dict:
                    - /path/to/distance_dict1.parquet
                    - /path/to/distance_dict2.parquet
                  year:
                    name: year_id
                    kernel: inverse
                    radius: [0.5, 1, 1.5]

        # Ensemble settings
        ensemble:
          groupby: [super_region_id]
          metric: rmse
          score: rover
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
    test: str = "test"
    id_subsets: dict[str, list[Any]] = {}
    groupby: list[str] = []

    rover_covsel: RoverCovselConfig | None = None
    spxmod: SPxModConfig | None = None
    weave: WeaveConfig | None = None
    ensemble: EnsembleConfig | None = None

    def model_post_init(self, *args, **kwargs) -> None:
        """Add global groupby attribute to stages."""
        for stage_config in [
            self.rover_covsel,
            self.spxmod,
            self.weave,
            self.ensemble,
        ]:
            if stage_config is not None:
                if isinstance(stage_config, WeaveConfig):
                    for model_config in stage_config.models.values():
                        self.add_groups(model_config)
                else:
                    self.add_groups(stage_config)

    def add_groups(self, stage_config: StageConfig):
        """Add global groups to stage."""
        for group in self.groupby:
            if group not in stage_config.groupby:
                stage_config.groupby.append(group)
