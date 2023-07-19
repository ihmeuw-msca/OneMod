from pathlib import Path
import sys

from onemod.pipeline.rover_stage import RoverStage


def test_rover_tasks(testing_tool, temporary_directory, sample_input_data, sample_config):

    # Create a set of rover tasks. Check that the correct commands are generated
    stage = RoverStage(testing_tool)

    tasks = stage.create_tasks(results_dir=temporary_directory, delete_intermediate=False,
                               cluster_name='dummy', rover_settings=sample_config['rover'],
                               input_data=sample_input_data)

    # Inspecting the settings, we are grouping by sex and age. 3 ages + 2 sexes = 6 tasks
    assert len(tasks) == 7  # 6 modeling tasks plus aggregation task
    expected_agg_task = tasks.pop()
    assert expected_agg_task.name == "rover_aggregation"
    assert len(expected_agg_task.upstream_tasks) == 6
    assert "aggregate_results rover" in expected_agg_task.command

    sample_model_task = tasks[0]
    assert not sample_model_task.upstream_tasks


def test_rover_tasks_with_deletion(testing_tool, temporary_directory,
                                   sample_input_data, sample_config):

    stage = RoverStage(testing_tool)

    tasks = stage.create_tasks(results_dir=temporary_directory, delete_intermediate=True,
                               cluster_name='dummy', rover_settings=sample_config['rover'],
                               input_data=sample_input_data)

    # Should be 13 tasks - 6 modeling, 1 agg, 6 deletion
    assert len(tasks) == 13

    aggregation_tasks = list(filter(lambda t: "aggregation" in t.name, tasks))
    deletion_tasks = list(filter(lambda t: "deletion" in t.name, tasks))
    modeling_tasks = list(filter(lambda t: "rover_model_task" in t.name, tasks))

    assert len(aggregation_tasks) == 1
    assert len(deletion_tasks) == 6
    assert len(modeling_tasks) == 6

    agg_task = aggregation_tasks[0]

    # Check both tasks have the same upstreams, the modeling tasks
    assert agg_task.upstream_tasks == set(modeling_tasks)
    assert deletion_tasks[0].upstream_tasks == {agg_task}
    assert modeling_tasks[0].upstream_tasks == set()


def test_batching():
    # TODO: Figure out a unit test to check if a large rover workflow can be chunked correctly
    assert True
