from onemod.orchestration.stage import StageTemplate


def test_swimr_tasks(testing_tool, temporary_directory, sample_config, sample_input_data):

    stage = StageTemplate(
        stage_name='swimr',
        config=sample_config,
        experiment_dir=temporary_directory,
        save_intermediate=True,
        resources_file=temporary_directory / 'resources.yml',
        tool=testing_tool,
        cluster_name='dummy'
    )
    tasks = stage.create_tasks([])

    # Breakdown:
    # Model 1 - 9 different parameters - 3 internal knots, 3 similarity multipliers
    # 6 subsets (3 age groups, 2 sex ids), 2 holdout sets, = 9*6*2 = 108 tasks

    # Model 2 - 9 parameters, 3 different thetas and 3 different intercept_thetas
    # 6 subsets, 2 holdoutsets = 108 tasks
    # Plus 1 aggregation task = 108 + 108 + 1 = 217
    assert len(tasks) == 217
