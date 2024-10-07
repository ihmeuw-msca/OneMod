from pathlib import Path
import pytest

from onemod import Pipeline
from onemod.config import PreprocessingConfig
from onemod.io import Input, Output
from onemod.stage import PreprocessingStage, KregStage, RoverStage, SpxmodStage
from onemod.types import Data

# TODO: should be env var at the least and point to shared dir for test assets
TEST_CONFIG_DIR = "tests/e2e/assets"

@pytest.mark.e2e
def test_e2e_onemod_example1_sequential(tmp_path):
    """
    End-to-end test for a the OneMod example1 pipeline.
    """
    # Define Stages
    preprocessing = PreprocessingStage(
        name="1_preprocessing",
        config=PreprocessingConfig(
            data=tmp_path / "data" / "input_data.parquet"
        ),
        input=Input(
            stage="1_preprocessing",
            required=dict(
                data=Data(
                    stage="1_preprocessing",
                    path=Path(tmp_path / "data" / "input_data.parquet"),
                    format="parquet"
                )
            )
        ),
        output=Output(
            stage="1_preprocessing",
            items=dict(
                data=Data(
                    stage="1_preprocessing",
                    path=Path(tmp_path / "data" / "preprocessed_data.parquet"),
                    format="parquet"
                )
            )   
        )
    )
    
    covariate_selection = RoverStage(
        name="2_covariate_selection",
        config=dict(cov_exploring=["cov1", "cov2", "cov3"]),
        groupby=["age_group_id"],
        # write_to_disk=True,  # TODO: implement
        input=Input(
            stage="2_covariate_selection",
            required=dict(
                data=Data(
                    stage="2_covariate_selection",
                    path=Path(tmp_path / "data" / "preprocessed_data.parquet"),
                    format="parquet"
                )
            )
        ),
        output=Output(
            stage="2_covariate_selection",
            items=dict(
                selected_covs=Data(
                    stage="2_covariate_selection",
                    path=Path(tmp_path / "data" / "selected_covs.parquet"),
                    format="parquet"
                )
            )
        )
    )

    global_model = SpxmodStage(
        name="3_global_model",
        # write_to_disk=True,  # TODO: implement
        input=Input(
            stage="3_global_model",
            required=dict(
                data=Data(
                    stage="3_global_model",
                    path=Path(tmp_path / "data" / "preprocessed_data.parquet"),
                    format="parquet"
                ),
                selected_covs=Data(
                    stage="3_global_model",
                    path=Path(tmp_path / "data" / "selected_covs.parquet"),
                    format="parquet"
                )
            )
        ),
        output=Output(
            stage="3_global_model",
            items=dict(
                predictions=Data(
                    stage="3_global_model",
                    path=Path(tmp_path / "data" / "global_predictions.parquet"),
                    format="parquet"
                )
            )
        )
    )

    location_model = SpxmodStage(
        name="4_location_model",
        groupby=["location_id"],
        # write_to_disk=True,  # TODO: implement
        input=Input(
            stage="4_location_model",
            required=dict(
                data=Data(
                    stage="4_location_model",
                    path=Path(tmp_path / "data" / "preprocessed_data.parquet"),
                    format="parquet"
                ),
                offset=Data(
                    stage="4_location_model",
                    path=Path(tmp_path / "data" / "global_predictions.parquet"),
                    format="parquet"
                )
            )
        ),
        output=Output(
            stage="4_location_model",
            items=dict(
                predictions=Data(
                    stage="4_location_model",
                    path=Path(tmp_path / "data" / "location_predictions.parquet"),
                    format="parquet"
                )
            )
        )
    )

    smoothing = KregStage(
        name="5_smoothing",
        groupby=["region_id"],
        # write_to_disk=True,  # TODO: implement
        input=Input(
            stage="5_smoothing",
            required=dict(
                data=Data(
                    stage="5_smoothing",
                    path=Path(tmp_path / "data" / "preprocessed_data.parquet"),
                    format="parquet"
                ),
                offset=Data(
                    stage="5_smoothing",
                    path=Path(tmp_path / "data" / "location_predictions.parquet"),
                    format="parquet"
                )
            )
        ),
        output=Output(
            stage="5_smoothing",
            items=dict(
                predictions=Data(
                    stage="5_smoothing",
                    path=Path(tmp_path / "data" / "smoothed_predictions.parquet"),
                    format="parquet"
                )
            )
        )
    )

    # Create pipeline
    dummy_pipeline = Pipeline(
        name="dummy_pipeline",
        config=dict(
            ids=["age_group_id", "location_id", "sex_id", "year_id"],
            mtype="binomial",
        ),
        directory=tmp_path,
        data=Path(tmp_path / "data" / "data.parquet"),
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
    preprocessing(data=Path(tmp_path / "data" / "input_data.parquet"))
    covariate_selection(data=preprocessing.output["data"])
    global_model(
        data=preprocessing.output["data"],
        selected_covs=covariate_selection.output["selected_covs"],
    )
    location_model(
        data=preprocessing.output["data"], offset=global_model.output["predictions"]
    )
    smoothing(
        data=preprocessing.output["data"],
        offset=location_model.output["predictions"],
    )

    # Execute stages in sequence
    dummy_pipeline.run(
        tool="sequential",  # TODO: update to whatever this arg is actually called
        config={}
    )
    
    # TODO: implement all of these or design different way to check statuseseses
    assert dummy_pipeline.stages["1_preprocessing"].status == "completed"
    assert dummy_pipeline.stages["2_covariate_selection"].status == "completed"
    assert dummy_pipeline.stages["3_global_model"].status == "completed"
    assert dummy_pipeline.stages["4_location_model"].status == "completed"
    assert dummy_pipeline.stages["5_smoothing"].status == "completed"
    assert dummy_pipeline.status == "completed"
