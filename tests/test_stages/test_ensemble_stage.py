from onemod.pipeline.ensemble_stage import EnsembleStage


def test_ensemble_tasks(testing_tool, temporary_directory, sample_config):

    stage = EnsembleStage(testing_tool)
    ensemble_settings = sample_config['ensemble']
    task = stage.create_tasks(
        results_dir=temporary_directory,
        cluster_name='dummy',
        ensemble_settings=ensemble_settings,
    )

    assert len(task) == 1
    assert "/bin/ensemble_model" in task[0].command
