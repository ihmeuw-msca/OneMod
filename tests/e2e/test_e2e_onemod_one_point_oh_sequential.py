import pytest

from onemod.constraints import bounds, no_inf
from onemod import Pipeline
from onemod.stage import PreprocessingStage, KregStage, RoverStage, SpxmodStage
from onemod.dtypes import Data, FilePath


@pytest.mark.e2e
@pytest.mark.skip(reason="Not yet implemented")
def test_e2e_onemod_one_point_oh_sequential():
    """End-to-end test for the OneMod 1.0 main branch pipeline."""
    pass
