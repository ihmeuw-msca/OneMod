from onemod.scheduler.scheduler import Scheduler


def test_rover_tasks(testing_tool, temporary_directory, sample_input_data, sample_config):

    # Create a set of rover tasks. Check that the correct commands are generated
    scheduler = Scheduler(config=sample_config, stages=['rover_covsel'])

    tasks = [
        scheduler.create_task(action) for action in scheduler.parent_action_generator()
    ]
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
