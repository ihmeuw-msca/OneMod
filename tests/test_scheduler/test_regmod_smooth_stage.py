from onemod.scheduler.scheduler import Scheduler
from onemod.scheduler.scheduling_utils import ParentTool, TaskRegistry


def test_rover_tasks(testing_tool, temporary_directory, sample_input_data, sample_config):

    ParentTool.tool = testing_tool
    TaskRegistry.registry.clear()

    # Create a set of regmod tasks. Check that the correct commands are generated
    scheduler = Scheduler(experiment_dir=temporary_directory, config=sample_config, stages=['regmod_smooth'])

    tasks = [
        scheduler.create_task(action) for action in scheduler.parent_action_generator()
    ]
    # 3 tasks - initialization, modeling and collection
    assert len(tasks) == 3

    init_task, model_task, agg_task = tasks

    assert agg_task.name == "collect_results"
    assert agg_task.upstream_tasks == {model_task}
    assert model_task.upstream_tasks == {init_task}
    assert init_task.upstream_tasks == set()
    assert "regmod_smooth_model" in model_task.command
    assert "collect_results --stage_name regmod_smooth" in agg_task.command
