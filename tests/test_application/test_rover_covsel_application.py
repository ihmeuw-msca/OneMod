from onemod.application.rover_covsel_application import RoverCovselApplication


def test_rover_covsel_application(temporary_directory, sample_input_data):
    # Check that the correct actions are generated.
    rover_covsel_app = RoverCovselApplication(temporary_directory)

    actions = list(rover_covsel_app.action_generator())
    assert len(actions) == 7  # 3 age groups * 2 sex ids = 6 + 1 collect results
    assert actions[0].name == "rover_covsel_model"
    assert actions[-1].name == "collect_results"
