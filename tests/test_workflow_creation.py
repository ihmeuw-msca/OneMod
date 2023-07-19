from onemod.pipeline.main import create_workflow


def test_rover_only_workflow(testing_tool, temporary_directory, sample_config, sample_input_data):
    workflow = create_workflow(
        directory=temporary_directory, models=['rover'], delete_intermediate=True,
        cluster_name='dummy', tool=testing_tool, configure_resources=False
    )
    # In total, we should have 3 age group ids, 2 sex ids = (3*2) = 6 modeling tasks
    # additionally, 1 aggregation and 6 deletion tasks means 13 total tasks
    assert len(workflow.tasks) == 13


def test_rover_weave_workflow(testing_tool, temporary_directory, sample_config, sample_input_data):

    workflow = create_workflow(
        directory=temporary_directory, models=['rover', 'weave', 'ensemble'],
        cluster_name='dummy', tool=testing_tool, configure_resources=False
    )

    # Rover stage should have 13 tasks as in prior unit test
    # Weave stage should have an additional 121 tasks based on the settings.
    assert len(workflow.tasks) == 135

    ensemble_model_task = workflow.get_tasks_by_node_args(
        task_template_name='ensemble_model_template'
    )[0]
    assert ensemble_model_task.name == "ensemble_task"
    assert ensemble_model_task.task_args['results_dir'] is not None

    # Weave tasks should have an upstream dependency on the rover tasks
    weave_model_tasks = workflow.get_tasks_by_node_args(
        task_template_name='weave_model_template'
    )
    sample_weave_task = weave_model_tasks[0]

    expected_rover_agg_task = sample_weave_task.upstream_tasks.pop()
    assert expected_rover_agg_task.name == "rover_aggregation"
    assert not sample_weave_task.upstream_tasks
