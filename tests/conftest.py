import itertools
from pathlib import Path

import numpy as np
import pytest
import os
import shutil
import yaml

from jobmon.client.api import Tool
import pandas as pd


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
    )
    data = pd.DataFrame(values,
                        columns=[
                            'super_region_id', 'location_id',
                            'sex_id', 'age_group_id', 'year_id'
                        ])

    # Generate false holdout columns
    data['holdout'] = 1
    data['holdout1'] = np.random.randint(0, 2, len(data))
    data['holdout2'] = np.random.randint(0, 2, len(data))

    # Save to the temp directory
    data_path = temporary_directory / 'data.parquet'
    data.to_parquet(data_path)

    # Update the data path key in the config with this value
    config['input_path'] = str(data_path)
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    return data


@pytest.fixture(scope='session')
def sample_config(temporary_directory):
    yaml_path = temporary_directory / 'config/settings.yml'
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
