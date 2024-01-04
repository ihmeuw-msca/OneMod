import pytest

from onemod.scheduler.scheduler import Scheduler


@pytest.mark.skip("E2E test working but weave collection task incompatible with toy data")
def test_end_to_end_local(testing_tool, temporary_directory, sample_input_data, sample_config):

    # Run the full application through rover, regmod, and weave.

    # Weave config creates way too many tasks. Trim down the config, only run model1
    sample_config.weave.models.pop("model2")

    scheduler = Scheduler(
        experiment_dir=temporary_directory,
        config=sample_config,
        stages=['rover_covsel', 'regmod_smooth', 'weave'],
    )

    # First run the schedule in memory
    scheduler.run(run_local=True)

    # Check for output files
    assert (temporary_directory / 'results' / 'rover_covsel' / 'summaries.csv').exists()
    assert (temporary_directory / 'results' / 'regmod_smooth' / 'predictions.parquet').exists()
    assert (temporary_directory / 'results' / 'weave' / 'predictions.parquet').exists()



@pytest.mark.skip("Extremely slow with Jobmon sequential distributor, weave collection task incompatible with toy data")
def test_end_to_end_remote(testing_tool, temporary_directory, sample_input_data, sample_config):

    # Run the full application through rover, regmod, and weave.
    # Weave config creates way too many tasks. Trim down the config, only run model1
    sample_config.weave.models.pop("model2")

    scheduler = Scheduler(
        experiment_dir=temporary_directory,
        config=sample_config,
        stages=['rover_covsel', 'regmod_smooth', 'weave'],
        default_cluster_name='sequential',
        resources_path=temporary_directory / 'config' / 'resources.yml',
        configure_resources=False,
    )

    # Run with Jobmon. Use the sequential executor so processes are actually executed
    # Should exit with "D" status.
    scheduler.run(run_local=False)

    # Check for output files
    assert (temporary_directory / 'results' / 'rover_covsel' / 'summaries.csv').exists()
    assert (temporary_directory / 'results' / 'regmod_smooth' / 'predictions.parquet').exists()
    # assert (temporary_directory / 'results' / 'weave' / 'predictions.parquet').exists()



