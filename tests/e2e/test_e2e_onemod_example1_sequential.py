from pathlib import Path

import pytest

from onemod import Pipeline
from onemod.config import PreprocessingConfig
from onemod.dtypes import Data
from onemod.stage import KregStage, PreprocessingStage, RoverStage, SpxmodStage


@pytest.mark.skip(
    reason="Test not implemented yet. Needs actual stages implemented and test data assets."
)
@pytest.mark.e2e
@pytest.mark.requires_data
def test_e2e_onemod_example1_sequential(small_input_data, test_base_dir):
    """
    End-to-end test for a the OneMod example1 pipeline.
    """

    # Define Stages
    preprocessing = PreprocessingStage(
        name="1_preprocessing",
        config=PreprocessingConfig(data=small_input_data),
        input_validation=dict(data=small_input_data),
        output_validation=dict(
            data=Data(
                stage="1_preprocessing",
                path=Path(test_base_dir, "data", "preprocessed_data.parquet"),
                format="parquet",
            )
        ),
    )

    covariate_selection = RoverStage(
        name="2_covariate_selection",
        config=dict(cov_exploring=["cov1", "cov2", "cov3"]),
        groupby=["age_group_id"],
        # write_to_disk=True,  # TODO: implement
        input_validation=dict(
            data=Data(
                stage="2_covariate_selection",
                path=Path(test_base_dir, "data", "preprocessed_data.parquet"),
                format="parquet",
            )
        ),
        output_validation=dict(
            selected_covs=Data(
                stage="2_covariate_selection",
                path=Path(test_base_dir, "data", "selected_covs.parquet"),
                format="parquet",
            )
        ),
    )

    global_model = SpxmodStage(
        name="3_global_model",
        # write_to_disk=True,  # TODO: implement
        input_validation=dict(
            data=Data(
                stage="3_global_model",
                path=Path(test_base_dir, "data", "preprocessed_data.parquet"),
                format="parquet",
            ),
            selected_covs=Data(
                stage="3_global_model",
                path=Path(test_base_dir, "data", "selected_covs.parquet"),
                format="parquet",
            ),
        ),
        output_validation=dict(
            predictions=Data(
                stage="3_global_model",
                path=Path(test_base_dir, "data", "global_predictions.parquet"),
                format="parquet",
            )
        ),
    )

    location_model = SpxmodStage(
        name="4_location_model",
        groupby=["location_id"],
        # write_to_disk=True,  # TODO: implement
        input_validation=dict(
            data=Data(
                stage="4_location_model",
                path=Path(test_base_dir, "data", "preprocessed_data.parquet"),
                format="parquet",
            ),
            offset=Data(
                stage="4_location_model",
                path=Path(test_base_dir, "data", "global_predictions.parquet"),
                format="parquet",
            ),
        ),
        output_validation=dict(
            predictions=Data(
                stage="4_location_model",
                path=Path(
                    test_base_dir, "data", "location_predictions.parquet"
                ),
                format="parquet",
            )
        ),
    )

    smoothing = KregStage(
        name="5_smoothing",
        groupby=["region_id"],
        # write_to_disk=True,  # TODO: implement
        input_validation=dict(
            data=Data(
                stage="5_smoothing",
                path=Path(test_base_dir, "data", "preprocessed_data.parquet"),
                format="parquet",
            ),
            offset=Data(
                stage="5_smoothing",
                path=Path(
                    test_base_dir, "data", "location_predictions.parquet"
                ),
                format="parquet",
            ),
        ),
        output_validation=dict(
            predictions=Data(
                stage="5_smoothing",
                path=Path(
                    test_base_dir, "data", "smoothed_predictions.parquet"
                ),
                format="parquet",
            )
        ),
    )

    # Create pipeline
    dummy_pipeline = Pipeline(
        name="dummy_pipeline",
        config=dict(
            ids=["age_group_id", "location_id", "sex_id", "year_id"],
            mtype="binomial",
        ),
        directory=test_base_dir,
        data=Path(test_base_dir, "data", "data.parquet"),
        groupby=["sex_id"],
    )

    # Add stages
    dummy_pipeline.add_stages(
        [
            preprocessing,
            covariate_selection,
            global_model,
            location_model,
            smoothing,
        ]
    )

    # Define dependencies
    preprocessing(data=Path(test_base_dir, "data", "input_data.parquet"))
    covariate_selection(data=preprocessing.output["data"])
    global_model(
        data=preprocessing.output["data"],
        selected_covs=covariate_selection.output["selected_covs"],
    )
    location_model(
        data=preprocessing.output["data"],
        offset=global_model.output["predictions"],
    )
    smoothing(
        data=preprocessing.output["data"],
        offset=location_model.output["predictions"],
    )

    # Execute stages in sequence
    dummy_pipeline.evaluate(backend="local")
