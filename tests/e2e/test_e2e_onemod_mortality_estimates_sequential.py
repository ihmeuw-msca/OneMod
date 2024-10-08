import pytest

from onemod.constraints import bounds, no_inf
from onemod import Pipeline
# from onemod.stage import PreprocessingStage, KregStage, RoverStage, SpxmodStage, WeaveStage, UncertaintyStage, EnsembleStage
from onemod.types import Data, FilePath

@pytest.mark.e2e
@pytest.mark.skip(reason="Not yet implemented")
def test_e2e_onemod_mortality_estimates_sequential(tmp_path):
    """
    End-to-end test for the OneMod 1.0 mortality estimates branch pipeline.
    
    From the mortality-estimates branch circa 2024-10-05:
    all_stages = [
        "rover_covsel",
        "spxmod",
        "weave",
        "kreg",
        "uncertainty",
        "ensemble",
    ]
    """
    # Define Stages
    preprocessing = PreprocessingStage(
        name="1_preprocessing",
        inputs=[FilePath],
        outputs=[
            Data.use_config(config_file="tests/e2e/config/1_preprocessing.json")
        ]
    )