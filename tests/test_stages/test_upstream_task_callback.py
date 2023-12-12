from onemod.scheduler.scheduling_utils import TaskRegistry, upstream_task_callback


def test_upstream_task_callback():
    """Test that the upstream_task_callback returns the correct tasks."""

    # Add some mock data to the task registry.
    TaskRegistry.put('initialize_results', 'initialize_results')
    for i in range(3):
        TaskRegistry.put('rover_covsel_model', f'rover_covsel_{i}')
    TaskRegistry.put('collect_results_rover_covsel', 'collect_results_rover_covsel')
    TaskRegistry.put('regmod_smooth_model', 'regmod_smooth_model')
    TaskRegistry.put('collect_results_regmod_smooth', 'collect_results_regmod_smooth')

    for i in range(4):
        TaskRegistry.put('weave_model', f'weave_{i}')

    TaskRegistry.put('collect_results_weave', 'collect_results_weave')

    # Check to see if the callbacks return the right things
    class MockAction:
        def __init__(self, name: str):
            self.name = name

    rover_covsel_action = MockAction('rover_covsel_model')
    rover_upstreams = upstream_task_callback(rover_covsel_action)
    assert rover_upstreams == ['initialize_results']

    rover_collection_action = MockAction('collect_results_rover_covsel')
    rover_collection_upstreams = upstream_task_callback(rover_collection_action)
    assert set(rover_collection_upstreams) == {"rover_covsel_0", "rover_covsel_1", "rover_covsel_2"}

    regmod_smooth_action = MockAction('regmod_smooth_model')
    regmod_upstreams = upstream_task_callback(regmod_smooth_action)
    assert set(regmod_upstreams) == {'initialize_results', 'collect_results_rover_covsel'}

    # Check to see if upstreams work in the case of partial stages.
    TaskRegistry.registry.clear()
    weave_action = MockAction('weave_model')
    weave_upstreams = upstream_task_callback(weave_action)
    assert weave_upstreams == []

    TaskRegistry.put("initialize_results", "initialize_results")
    weave_upstreams = upstream_task_callback(weave_action)
    assert weave_upstreams == ["initialize_results"]
