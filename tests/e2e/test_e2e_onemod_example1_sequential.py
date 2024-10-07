from pathlib import Path
import pytest

from onemod.constraints import bounds, no_inf
from onemod import Pipeline
from onemod.stage import PreprocessingStage, KregStage, RoverStage, SpxmodStage
from onemod.types import Data, FilePath

# TODO: should be env var at the least and point to shared dir for test assets
TEST_CONFIG_DIR = "tests/e2e/assets"

@pytest.mark.e2e
@pytest.mark.skip(reason="Not yet implemented")
def test_e2e_onemod_mortality_estimates_sequential(tmp_path):
    """
    End-to-end test for a the OneMod example1 pipeline.
    """
    # Define Stages
    preprocessing = PreprocessingStage(
        name="1_preprocessing",
        config={},
        inputs=dict(data_path=FilePath),
        outputs=dict(
            data=Data.from_config(config_file=Path(TEST_CONFIG_DIR / "config" / "1_preprocessing.json"))
        )
    )
    
    covariate_selection = RoverStage(
        name="2_covariate_selection",
        config=dict(cov_exploring=["cov1", "cov2", "cov3"]),
        groupby=["age_group_id"],
        write_to_disk=True,
        inputs=dict(
            data=Data.use_validation(
                columns={},
                shape=None,
            )
        ),
        outputs=dict(
            Data.use_validation(...)
        )
    )

    global_model = SpxmodStage(
        name="3_global_model",
        write_to_disk=True,
        inputs=dict(
            Data.use_validation(...)
        ),
        outputs=dict(
            Data.use_validation(...)
        )
    )

    location_model = SpxmodStage(
        name="4_location_model",
        groupby=["location_id"],
        write_to_disk=True,
        inputs=dict(
            Data.use_validation(...)
        ),
        outputs=dict(
            Data.use_validation(...)
        )
    )

    smoothing = KregStage(
        name="5_smoothing",
        groupby=["region_id"],
        write_to_disk=True,
        inputs=dict(
            Data.use_validation(...)
        ),
        outputs=dict(
            Data.use_validation(...)
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
    preprocessing(data=Path(tmp_path / "data" / "data.parquet"))
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
        how="sequential",  # TODO: update to whatever this arg is actually called
        config={}
    )
    
    # TODO: implement all of these or design different way to check statuseseses
    assert dummy_pipeline.stages["1_preprocessing"].status == "completed"
    assert dummy_pipeline.stages["2_covariate_selection"].status == "completed"
    assert dummy_pipeline.stages["3_global_model"].status == "completed"
    assert dummy_pipeline.stages["4_location_model"].status == "completed"
    assert dummy_pipeline.stages["5_smoothing"].status == "completed"
    assert dummy_pipeline.status == "completed"
