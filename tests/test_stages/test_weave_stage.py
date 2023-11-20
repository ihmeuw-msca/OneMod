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
    # Breakdown: Model1 has 3 parameters - age, year, and year
    # (in the settings.weave.models.model1.dimensions register).

    # Each of the parameters has 3 unique values to iterate over; age and location have 3 values
    # for radius and year has 3 values for exponent. 3*3*3 = 27 parameters.

    # There are 2 unique combinations of the groupby parameters (sex, super region)
    # and 2 holdout columns plus a "full" holdout set.
    # 27 * 2 * 3 = 162, and no batching multiplier means we have 162 tasks total for model1.

    # Model2 has 2 parameters, year and location, and year has 3 values of exponent to form
    # submodels from, making 3 * 1 = 3 parameter sets.

    # As in model1 we have 2 subsets and 3 holdout columns, making 6 subsets in total, and 3
    # holdout folds
    # 3 * 6 * 3 = 54 tasks for model2.

    # Additionally, since we have a max batch size of 3, we have to add an additional factor
    # of 2 since each subset has to be split into two batches (6 // 3 = 2)

    # 162 + 54 * 2 = 216, plus 1 aggregation task makes 217
    assert len(tasks) == 271

