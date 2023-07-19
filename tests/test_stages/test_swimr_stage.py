from onemod.pipeline.swimr_stage import SwimrStage


def test_swimr_tasks(testing_tool, temporary_directory, sample_config, sample_input_data):

    stage = SwimrStage(testing_tool)
    swimr_settings = sample_config['swimr']
    holdout_cols = sample_config['col_holdout']
    tasks = stage.create_tasks(
        results_dir=temporary_directory,
        cluster_name='dummy',
        input_data=sample_input_data,
        swimr_settings=swimr_settings,
        holdout_cols=holdout_cols,
    )

    # Breakdown:
    # Model 1 - 9 different parameters - 3 internal knots, 3 similarity multipliers
    # 6 subsets (3 age groups, 2 sex ids), 2 holdout sets, = 9*6*2 = 108 tasks

    # Model 2 - 9 parameters, 3 different thetas and 3 different intercept_thetas
    # 6 subsets, 2 holdoutsets = 108 tasks
    # Plus 1 aggregation task = 108 + 108 + 1 = 217
    assert len(tasks) == 217
