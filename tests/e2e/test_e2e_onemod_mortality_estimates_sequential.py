import pytest


@pytest.mark.e2e
@pytest.mark.requires_data
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
    pass
