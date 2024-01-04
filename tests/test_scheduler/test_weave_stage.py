from onemod.scheduler.scheduler import Scheduler
from onemod.scheduler.scheduling_utils import ParentTool, TaskRegistry


def test_rover_tasks(testing_tool, temporary_directory, sample_input_data, sample_config):

    ParentTool.tool = testing_tool
    TaskRegistry.registry.clear()

    # Create a set of rover tasks. Check that the correct commands are generated
    scheduler = Scheduler(experiment_dir=temporary_directory, config=sample_config, stages=['weave'])

    tasks = [
        scheduler.create_task(action) for action in scheduler.parent_action_generator()
    ]
    # Breakdown: Model1 has 2 parameters - age and location
    # (in the settings.weave.models.model1.dimensions register).

    # Age has 1 value of radius to iterate over, and location has 2.
    # 1*2 = 2 parameters.

    # There are 2 unique combinations of the groupby parameters (sex, super region)
    # and 2 holdout columns plus a "full" holdout set.
    # 2 * 2 * 3 = 12, and no batching multiplier means we have 12 tasks total for model1.

    # Model2 has 1 parameter, year, with 1 value of exponent to form submodels from
    # The groupby parameter is now age and sex. With 3 ages and 2 sexes, we have 6 submodels.

    # As in model1 we have 3 holdout columns, making 18 subsets in total (6 subsets * 3 folds)
    # 1 * 18 = 18 tasks for model2.

    # Additionally, we have a max batch size of 6. In conftest.py, we have 3 rows per group
    # to account for the various holdout sets, meaning we will an additional factor of 3.
    # (18 tasks // 12 batch size = 2)

    # This means we have
    # 12 + 18 * 2 = 48, plus 1 aggregation task and 1 initialization makes 50
    assert len(tasks) == 50
