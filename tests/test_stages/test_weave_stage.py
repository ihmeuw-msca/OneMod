from onemod.orchestration.stage import StageTemplate


def test_weave_tasks(testing_tool, temporary_directory, sample_config, sample_input_data):

    stage = StageTemplate(
        stage_name='weave',
        config=sample_config,
        experiment_dir=temporary_directory,
        save_intermediate=True,
        resources_file=temporary_directory / 'resources.yml',
        tool=testing_tool,
        cluster_name='dummy'
    )
    tasks = stage.create_tasks([])
    # We expect 81 tasks.
    # Breakdown: Model1 has 2 year_id parameters and 2 location parameters, for 4 total
    # There are 6 unique combinations of the groupby parameters (sex, super region, age)
    # and 2 holdout columns.
    # 4 *6*2 = 48, and no batching multiplier means we have 24 tasks total.

    # Model2 has 2 year parameters, 2 unique combinations of the groupby parameter sex_id,
    # and 2 holdout columns.
    # There are 12 rows per sex, and a batch size of 3 means we'll have 4 batches
    # 2 * 2 * 4 * 2 = 32 tasks

    # 48 + 32 = 80, plus 1 aggregation task makes 41
    assert len(tasks) == 81

