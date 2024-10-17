import io
from contextlib import redirect_stdout
from pathlib import Path
import json

import pytest

from onemod import Pipeline
from onemod.constraints import Constraint
from onemod.dtypes import ColumnSpec, Data
from onemod.stage import PreprocessingStage, RoverStage, SpxmodStage, KregStage

from examples.custom_stage import CustomStage

from tests.utils import assert_equal_unordered


@pytest.fixture(scope="module")
def test_base_dir(tmp_path_factory):
    test_base_dir = tmp_path_factory.mktemp("example")
    return test_base_dir


@pytest.fixture(scope="module")
def test_input_data(test_assets_dir):
    test_input_data_path = Path(
        test_assets_dir, "e2e", "example1", "data", "small_data.parquet"
    )
    return test_input_data_path


@pytest.mark.e2e
def test_dummy_pipeline_sequential_run(test_base_dir, test_input_data):
    """End-to-end test for a the OneMod example pipeline with arbitrary configs and constraints, test data."""
    preprocessing = PreprocessingStage(
        name="preprocessing",
        config={},
        input_validation={
            "data": Data(
                stage="input_data",
                path=test_input_data,
                format="parquet",
                shape=(5, 52),
                columns={
                    "adult_hiv_death_rate": ColumnSpec(
                        type="float",
                        constraints=[
                            Constraint(
                                name="bounds",
                                args={"ge": 0, "le": 1}
                            )
                        ]
                    ),
                }
            )
        }
    )
    covariate_selection = RoverStage(
        name="covariate_selection",
        config={"cov_exploring": ["cov1", "cov2", "cov3"]},
        groupby=["age_group_id"],
    )
    global_model = SpxmodStage(
        name="global_model",
        config={"xmodel": {"variables": [{"name": "var1"}, {"name": "var2"}]}},
    )
    location_model = SpxmodStage(
        name="location_model",
        config={"xmodel": {"variables": [{"name": "var1"}, {"name": "var2"}]}},
        groupby=["location_id"],
    )
    smoothing = KregStage(
        name="smoothing",
        config={
            "kreg_model": {
                "age_scale": 1,
                "gamma_age": 1,
                "gamma_year": 1,
                "exp_location": 1,
                "lam": 1,
                "nugget": 1,
            },
            "kreg_fit": {
                "gtol": 1,
                "max_iter": 1,
                "cg_maxiter": 1,
                "cg_maxiter_increment": 1,
                "nystroem_rank": 1,
            },
        },
        groupby=["region_id"],
    )
    custom_stage = CustomStage(
        name="custom_stage",
        config={"custom_param": [1, 2]},
        groupby=["super_region_id"],
    )

    # Create pipeline
    dummy_pipeline = Pipeline(
        name="dummy_pipeline",
        config={
            "id_columns": ["age_group_id", "location_id", "sex_id", "year_id"],
            "model_type": "binomial",
        },
        directory=test_base_dir,
        data=test_input_data,
        groupby=["sex_id"],
    )

    # Add stages
    dummy_pipeline.add_stages(
        [
            preprocessing,
            covariate_selection,
            global_model,
            location_model,
            smoothing,
            custom_stage,
        ]
    )

    # Define dependencies
    preprocessing(data=dummy_pipeline.data)
    covariate_selection(data=preprocessing.output["data"])
    global_model(
        data=preprocessing.output["data"],
        selected_covs=covariate_selection.output["selected_covs"],
    )
    location_model(
        data=preprocessing.output["data"],
        offset=global_model.output["predictions"],
    )
    smoothing(
        data=preprocessing.output["data"],
        offset=location_model.output["predictions"],
    )
    custom_stage(
        observations=preprocessing.output["data"],
        predictions=smoothing.output["predictions"],
    )
    
    # Validate, build, and save the pipeline
    pipeline_json_path = test_base_dir / f"{dummy_pipeline.name}.json"
    dummy_pipeline.build()  # Saves to pipeline_json_path by default
    
    # Read in built pipeline representation
    with open(pipeline_json_path, "r") as f:
        dummy_pipeline_dict = json.load(f)
    
    assert dummy_pipeline_dict["name"] == 'dummy_pipeline'
    assert_equal_unordered(
        dummy_pipeline_dict["config"],
        {
            'id_columns': ['age_group_id', 'location_id', 'year_id', 'sex_id'],
            'model_type': 'binomial',
            'observation_column': 'obs',
            'prediction_column': 'pred',
            'weight_column': 'weights',
            'test_column': 'test',
            'holdout_columns': [],
            'coef_bounds': {}
        }
    )
    assert dummy_pipeline_dict["directory"] == str(test_base_dir)
    assert dummy_pipeline_dict["data"] == str(test_input_data)
    assert dummy_pipeline_dict["groupby"] == ['sex_id']
    assert_equal_unordered(
        dummy_pipeline_dict["dependencies"],
        {
            'preprocessing': [],
            'covariate_selection': ['preprocessing'],
            'global_model': ['covariate_selection', 'preprocessing'],
            'location_model': ['preprocessing', 'global_model'],
            'smoothing': ['preprocessing', 'location_model'],
            'custom_stage': ['smoothing', 'preprocessing']
        }
    )
    
    # Run the pipeline (local backend)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        dummy_pipeline.evaluate(backend="local", method="run")
        
    actual_output = buffer.getvalue().strip()
    
    sample_expected_outputs = ["running preprocessing", "running covariate_selection", "running global_model", "running location_model", "running smoothing", "running custom_stage"]
    
    for expected_output in sample_expected_outputs:
        assert expected_output in actual_output
