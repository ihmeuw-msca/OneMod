from onemod.scheduler.scheduler import Scheduler
from onemod.scheduler.scheduling_utils import ParentTool


def test_rover_tasks(testing_tool, temporary_directory, sample_input_data, sample_config):

    ParentTool.tool = testing_tool

    # Create a set of rover tasks. Check that the correct commands are generated
    scheduler = Scheduler(experiment_dir=temporary_directory, config=sample_config, stages=['rover_covsel'])

    tasks = [
        scheduler.create_task(action) for action in scheduler.parent_action_generator()
    ]
    # Inspecting the settings, we are grouping by sex and age. 3 ages + 2 sexes = 6 tasks
    assert len(tasks) == 8  # 6 modeling tasks plus init and aggregation task
    expected_agg_task = tasks.pop()
    assert expected_agg_task.name == "collect_results_rover_covsel"
    assert len(expected_agg_task.upstream_tasks) == 6
    assert "collect_results --stage_name rover_covsel" in expected_agg_task.command

    init_task = tasks[0]
    assert init_task.name == "initialize_results"
    assert len(init_task.upstream_tasks) == 0

    sample_model_task = tasks[-1]
    assert sample_model_task.upstream_tasks == {init_task}  # One initialization task as upstream


def test_batching():
    # TODO: Figure out a unit test to check if a large rover workflow can be chunked correctly
    assert True
