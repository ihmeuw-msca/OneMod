import pytest

from onemod.main import create_workflow


@pytest.mark.parametrize(
    'stages,expected_tasks',
    [
        (['rover_covsel'], 8),
        (['rover_covsel', 'regmod_smooth'], 10),
        (['weave', 272]),
        (['regmod_smooth', 'weave'], 274)
    ]
)
def test_workflow_stage_creation(
    testing_tool, temporary_directory, sample_config,  # pytest fixtures
    stages, expected_tasks  # test parameters
):
    workflow = create_workflow(
        directory=temporary_directory, stages=stages, save_intermediate=True,
        cluster_name='dummy', tool=testing_tool, configure_resources=False,
        config=sample_config
    )
    # Should have however many modeling tasks as determined by subsets,
    # plus a global +1 initialization task and aggregation tasks for each stage.
    assert len(workflow.tasks) == expected_tasks
