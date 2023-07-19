from onemod.pipeline.weave_stage import WeaveStage


def test_weave_tasks(testing_tool, temporary_directory, sample_config, sample_input_data):

    stage = WeaveStage(testing_tool)
    weave_settings = sample_config['weave']
    holdout_cols = sample_config['col_holdout']
    tasks = stage.create_tasks(
        results_dir=temporary_directory,
        cluster_name='dummy',
        input_data=sample_input_data,
        weave_settings=weave_settings,
        holdout_cols=holdout_cols
    )
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

