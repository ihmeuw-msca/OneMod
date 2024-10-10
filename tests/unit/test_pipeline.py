import pytest

from onemod.config import Config
from onemod.pipeline import Pipeline
from onemod.stage import Stage


class DummyStage(Stage):
    config: Config
    _required_input: set[str] = {"data.parquet", "covariates.csv"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}
    
@pytest.fixture(scope="module")
def example_base_dir(tmp_path_factory):
    example_base_dir = tmp_path_factory.mktemp("example")
    return example_base_dir

@pytest.fixture(scope="module")
def stage_1(example_base_dir):
    stage_1 = DummyStage(
        name="stage_1",
        directory=example_base_dir / "stage_1",
        config={},
    )
    stage_1.directory = example_base_dir / "stage_1"
    stage_1(
        data=example_base_dir / "stage_1" / "data.parquet",
        covariates=example_base_dir / "stage_1" / "covariates.csv"
    )
    return stage_1

@pytest.fixture(scope="module")
def stage_2(example_base_dir, stage_1):
    stage_2 = DummyStage(
        name="stage_2",
        directory=example_base_dir / "stage_2",
        config={},
    )
    stage_2.directory = example_base_dir / "stage_2"
    stage_2(
        data=stage_1.output["predictions"],
        covariates="/path/to/covariates.csv"
    )
    return stage_2

@pytest.mark.unit
def test_pipeline_initialization(example_base_dir):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=example_base_dir)
    assert pipeline.name == "test_pipeline"
    assert pipeline.stages == {}
    assert pipeline.dependencies == {}
    
@pytest.mark.integration
def test_add_stage_without_dependencies(example_base_dir):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=example_base_dir)
    stage = Stage(name="stage_a", config={})
    pipeline.add_stage(stage)
    assert "stage_a" in pipeline.stages
    assert pipeline.dependencies["stage_a"] == set()

@pytest.mark.integration
def test_add_stages_with_dependencies(example_base_dir, stage_1, stage_2):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=example_base_dir)
    pipeline.add_stage(stage_1)  # TODO: AttributeError: 'Config' object has no attribute 'update'
    pipeline.add_stage(stage_2)  # TODO: AttributeError: 'Config' object has no attribute 'update'
    
    assert "stage_a" in pipeline.stages
    assert "stage_b" in pipeline.stages
    assert pipeline.dependencies["stage_b"] == {"stage_a"}

@pytest.mark.integration
def test_duplicate_stage_error(example_base_dir):
    pipeline = Pipeline(name="duplicate_stage_pipeline", config={"ids": []}, directory=example_base_dir)
    stage_a = Stage(name="stage_a", config={})
    pipeline.add_stage(stage_a)
    with pytest.raises(ValueError, match="stage 'stage_a' already exists"):
        pipeline.add_stage(stage_a)  # Add the same stage again

@pytest.mark.integration
def test_pipeline_with_no_stages(example_base_dir):
    pipeline = Pipeline(name="empty_pipeline", config={"ids": []}, directory=example_base_dir)
    execution_order = pipeline.get_execution_order()
    assert execution_order == []

@pytest.mark.integration
def test_pipeline_with_single_stage(example_base_dir):
    pipeline = Pipeline(name="single_stage_pipeline", config={"ids": []}, directory=example_base_dir)
    stage = Stage(name="stage_a", config={})
    pipeline.add_stage(stage)
    execution_order = pipeline.get_execution_order()
    assert execution_order == ["stage_a"]

@pytest.mark.integration
def test_pipeline_with_valid_dependencies(example_base_dir):
    pipeline = Pipeline(name="valid_pipeline", config={"ids": []}, directory=example_base_dir)
    stage_a = Stage(name="stage_a", config={})
    stage_b = Stage(name="stage_b", config={})
    stage_c = Stage(name="stage_c", config={})
    pipeline.add_stage(stage_a)
    pipeline.add_stage(stage_b)
    pipeline.add_stage(stage_c)
    
    stage_a(data=example_base_dir / "stage_a" / "data.parquet")
    stage_b(data=stage_a.output["data"])
    stage_c(data=stage_a.output["data"], selected_covs=stage_b.output["selected_covs"])
    
    execution_order = pipeline.get_execution_order()
    assert execution_order == ["stage_a", "stage_b", "stage_c"]

@pytest.mark.integration
def test_pipeline_with_cyclic_dependencies(example_base_dir):
    pipeline = Pipeline(name="cyclic_pipeline", config={"ids": []}, directory=example_base_dir)
    stage_a = Stage(name="stage_a", config={})
    stage_b = Stage(name="stage_b", config={})
    stage_c = Stage(name="stage_c", config={})
    pipeline.add_stage(stage_a)
    pipeline.add_stage(stage_b)
    pipeline.add_stage(stage_c)
    
    with pytest.raises(ValueError, match="Cycle detected"):
        pipeline.get_execution_order()

@pytest.mark.integration
def test_pipeline_from_json_with_undefined_dependencies(example_base_dir):
    pipeline_dir = example_base_dir / "invalid_pipeline"
    pipeline_json = {
        "name": "invalid_pipeline",
        "config": {},
        "directory": str(pipeline_dir),
        "stages": [
            {"name": "stage_b", "config": {}}
        ],
        "dependencies": {
            "stage_b": ["stage_a"]
        }
    }
    with pytest.raises(ValueError, match="Dependency 'stage_a' not found in pipeline."):
        pipeline = Pipeline.from_json(pipeline_json)

@pytest.mark.integration
def test_build_dag_no_stages(example_base_dir):
    pipeline = Pipeline(name="empty_pipeline", config={"ids": []}, directory=example_base_dir)
    dag = pipeline.build_dag()
    assert dag == {}

@pytest.mark.integration
def test_build_dag_multiple_stages(example_base_dir):
    pipeline = Pipeline(name="multi_stage_pipeline", config={"ids": []}, directory=example_base_dir)
    stage_a = Stage(name="stage_a", config={})
    stage_b = Stage(name="stage_b", config={})
    pipeline.add_stage(stage_a)
    pipeline.add_stage(stage_b)
    stage_b(data=stage_a.output["data"])
    
    dag = pipeline.build_dag()
    expected_dag = {
        "stage_b": ["stage_a"],
        "stage_a": []
    }
    assert dag == expected_dag

@pytest.mark.integration
def test_validate_dag_with_undefined_dependency(example_base_dir):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=example_base_dir)
    stage_a = Stage(name="stage_a", config={})
    stage_b = Stage(name="stage_b", config={})
    pipeline.add_stage(stage_a)
    pipeline.add_stage(stage_b)
    pipeline.dependencies["stage_b"] = ["stage_c"]  # Undefined stage_c
    
    with pytest.raises(ValueError, match="Stage 'stage_c' is not defined"):
        pipeline.validate_dag()

@pytest.mark.integration
def test_validate_dag_with_self_dependency(example_base_dir):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=example_base_dir)
    stage_a = Stage(name="stage_a", config={})
    pipeline.add_stage(stage_a)
    stage_a(data=stage_a.output["data"])  # Self-dependency

    with pytest.raises(ValueError, match="Stage 'stage_a' cannot depend on itself"):
        pipeline.validate_dag()
