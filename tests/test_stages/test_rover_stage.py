from onemod.orchestration.stage import StageTemplate


def test_rover_tasks(testing_tool, temporary_directory, sample_input_data, sample_config):

    # Create a set of rover tasks. Check that the correct commands are generated
    stage = StageTemplate(
        stage_name='rover_covsel',
        config=sample_config,
        experiment_dir=temporary_directory,
        save_intermediate=True,
        resources_file=temporary_directory / 'resources.yml',
        tool=testing_tool,
        cluster_name='dummy'
    )

    tasks = stage.create_tasks(upstream_tasks=[])
    # Inspecting the settings, we are grouping by sex and age. 3 ages + 2 sexes = 6 tasks
    assert len(tasks) == 7  # 6 modeling tasks plus aggregation task
    expected_agg_task = tasks.pop()
    assert expected_agg_task.name == "rover_covsel_collection_task"
    assert len(expected_agg_task.upstream_tasks) == 6
    assert "collect_results" in expected_agg_task.command
    assert "--stage_name rover_covsel" in expected_agg_task.command

    sample_model_task = tasks[0]
    assert not sample_model_task.upstream_tasks


def test_batching():
    # TODO: Figure out a unit test to check if a large rover workflow can be chunked correctly
    assert True
