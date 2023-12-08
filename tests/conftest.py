import itertools
from pathlib import Path

from jobmon.client.api import Tool
import numpy as np
import pandas as pd
import pytest
import os
import shutil
import yaml


from onemod.schema.models.api import OneModConfig


@pytest.fixture(scope='session')
def testing_tool():

    tool = Tool(name="onemod_unit_testing_tool")
    tool.set_default_cluster_name('dummy')
    return tool


@pytest.fixture(scope='session')
def temporary_directory(tmp_path_factory):

    # Create a temporary directory and copy the settings file there
    current_dir = Path(__file__).resolve().parent
    settings_file = current_dir / 'settings.yml'
    resource_file = current_dir / 'resources.yml'

    results_path = tmp_path_factory.mktemp('results', numbered=False)
    tmp_path = results_path.parent
    os.mkdir(tmp_path / 'config')
    shutil.copy(settings_file, tmp_path / 'config')
    shutil.copy(resource_file, tmp_path / 'config')
    return tmp_path


@pytest.fixture(scope='session')
def sample_input_data(temporary_directory):
    # Write some example data to the temp directory.
    # For now, only useful for groupbys on the keys
    yaml_path = temporary_directory / 'config/settings.yml'
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    super_region_id = [1]
    location_ids = config['location_id']
    sex_ids = config['sex_id']
    age_group_ids = config['age_group_id']
    year_ids = config['year_id']

    values = list(itertools.product(
        super_region_id, location_ids, sex_ids, age_group_ids, year_ids)
    ) * 3  # Generate at least 3 rows per group
    data = pd.DataFrame(
        values,
        columns=[
            'super_region_id', 'location_id',
            'sex_id', 'age_group_id', 'year_id'
        ]
    )

    # Mock an age mid column
    data['age_mid'] = data['age_group_id']

    # Generate false holdout columns.
    # Need at least one holdout and one non holdout row per group.
    num_params = len(values) // 3
    data['holdout1'] = [0] * num_params + [1] * num_params * 2
    data['holdout2'] = data['holdout1']
    data['test'] = [0] * num_params * 2 + [1] * num_params

    # Generate fake covariate columns
    data['cov1'] = np.random.rand(len(data))
    data['cov2'] = np.random.rand(len(data))
    data['cov3'] = np.random.rand(len(data))

    # Generate an observations column, random from 0 to 1
    data['obs_rate'] = np.random.rand(len(data))

    # Save to the temp directory
    os.mkdir(temporary_directory / 'data')
    data_path = temporary_directory / 'data' / 'data.parquet'
    data.to_parquet(data_path)

    # Update the data path key in the config with this value
    config['input_path'] = str(data_path)
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    return data


@pytest.fixture(scope='session')
def sample_config_file(temporary_directory, sample_input_data):
    yaml_path = temporary_directory / 'config/settings.yml'
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture(scope='session')
def sample_config(sample_config_file):
    config = OneModConfig(**sample_config_file)
    config.rover_covsel.inherit()
    config.regmod_smooth.inherit()
    config.weave.inherit()
    config.swimr.inherit()
    config.ensemble.inherit()
    return config
