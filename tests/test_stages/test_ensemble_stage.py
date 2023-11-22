import pytest

from onemod.orchestration.stage import StageTemplate


@pytest.mark.skip("Not implemented yet")
def test_ensemble_tasks(testing_tool, temporary_directory, sample_config):

    stage = StageTemplate(
        stage_name='ensemble',
        config=sample_config,
        experiment_dir=temporary_directory,
        save_intermediate=True,
        resources_file=temporary_directory / 'resources.yml',
        tool=testing_tool,
        cluster_name='dummy'
    )
    task = stage.create_tasks([])

    assert len(task) == 1
    assert "/bin/ensemble_model" in task[0].command
