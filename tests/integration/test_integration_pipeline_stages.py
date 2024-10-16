import pytest

from onemod.config import StageConfig
from onemod.pipeline import Pipeline
from onemod.stage import Stage


class DummyStage(Stage):
    config: StageConfig
    _required_input: set[str] = {"data.parquet"}
    _optional_input: set[str] = {"priors.pkl"}
    _output: set[str] = {"predictions.parquet", "model.pkl"}
    
@pytest.fixture(scope="module")
def test_base_dir(tmp_path_factory):
    test_base_dir = tmp_path_factory.mktemp("example")
    return test_base_dir

@pytest.fixture(scope="module")
def stage_1(test_base_dir):
    stage_1 = DummyStage(
        name="stage_1",
        directory=test_base_dir / "stage_1",
        config={},
    )
    stage_1.directory = test_base_dir / "stage_1"
    stage_1(
        data=test_base_dir / "stage_1" / "data.parquet",
        covariates=test_base_dir / "stage_1" / "covariates.csv"
    )
    return stage_1

@pytest.fixture(scope="module")
def stage_2(test_base_dir, stage_1):
    stage_2 = DummyStage(
        name="stage_2",
        directory=test_base_dir / "stage_2",
        config={},
    )
    stage_2.directory = test_base_dir / "stage_2"
    stage_2(
        data=stage_1.output["predictions"],
        covariates="/path/to/covariates.csv"
    )
    return stage_2

@pytest.mark.unit
def test_pipeline_initialization(test_base_dir):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=test_base_dir)
    assert pipeline.name == "test_pipeline"
    assert pipeline.stages == {}
    assert pipeline.dependencies == {}
    
@pytest.mark.integration
def test_add_stage_without_dependencies(test_base_dir):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=test_base_dir)
    stage = DummyStage(name="stage_1", config={})
    pipeline.add_stage(stage)
    assert "stage_1" in pipeline.stages
    assert pipeline.dependencies["stage_1"] == set()

@pytest.mark.integration
def test_add_stages_with_dependencies(test_base_dir, stage_1, stage_2):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=test_base_dir)
    pipeline.add_stage(stage_1)
    pipeline.add_stage(stage_2)
    
    assert "stage_1" in pipeline.stages
    assert "stage_2" in pipeline.stages
    assert pipeline.dependencies["stage_2"] == {"stage_1"}

@pytest.mark.integration
def test_duplicate_stage_error(test_base_dir):
    pipeline = Pipeline(name="duplicate_stage_pipeline", config={"ids": []}, directory=test_base_dir)
    stage_1 = Stage(name="stage_1", config={})
    pipeline.add_stage(stage_1)
    with pytest.raises(ValueError, match="stage 'stage_1' already exists"):
        pipeline.add_stage(stage_1)  # Add the same stage again

@pytest.mark.integration
def test_pipeline_with_no_stages(test_base_dir):
    pipeline = Pipeline(name="empty_pipeline", config={"ids": []}, directory=test_base_dir)
    execution_order = pipeline.get_execution_order()
    assert execution_order == []

@pytest.mark.integration
def test_pipeline_with_single_stage(test_base_dir):
    pipeline = Pipeline(name="single_stage_pipeline", config={"ids": []}, directory=test_base_dir)
    stage = Stage(name="stage_1", config={})
    pipeline.add_stage(stage)
    execution_order = pipeline.get_execution_order()
    assert execution_order == ["stage_1"]

@pytest.mark.integration
def test_pipeline_with_valid_dependencies(test_base_dir):
    pipeline = Pipeline(name="valid_pipeline", config={"ids": []}, directory=test_base_dir)
    stage_1 = DummyStage(name="stage_1", config={})
    stage_2 = DummyStage(name="stage_2", config={})
    stage_3 = DummyStage(name="stage_3", config={})
    pipeline.add_stage(stage_1)
    pipeline.add_stage(stage_2)
    pipeline.add_stage(stage_3)
    
    stage_1(data=test_base_dir / "stage_1" / "data.parquet")
    stage_2(data=stage_1.output["predictions"])
    stage_3(data=stage_1.output["predictions"], selected_covs=stage_2.output["predictions"])
    
    execution_order = pipeline.get_execution_order()
    assert execution_order == ["stage_1", "stage_2", "stage_3"]

@pytest.mark.skip(reason="Test not yet implemented.")
@pytest.mark.integration
def test_pipeline_with_cyclic_dependencies(test_base_dir):
    pipeline = Pipeline(name="cyclic_pipeline", config={"ids": []}, directory=test_base_dir)
    stage_1 = DummyStage(name="stage_1", config={})
    stage_2 = DummyStage(name="stage_2", config={})
    stage_3 = DummyStage(name="stage_3", config={})
    pipeline.add_stage(stage_1)
    pipeline.add_stage(stage_2)
    pipeline.add_stage(stage_3)
    
    with pytest.raises(ValueError, match="Cycle detected"):
        pipeline.get_execution_order()

@pytest.mark.skip(reason="Test not yet implemented.")
@pytest.mark.integration
def test_pipeline_with_undefined_dependencies(test_base_dir):
    pipeline_dir = test_base_dir / "invalid_pipeline"
    invalid_pipeline = Pipeline(
        name="invalid_pipeline",
        config={"ids": []},
        directory=pipeline_dir
    )
    
    stage_2 = DummyStage(name="stage_2", config={})
    invalid_pipeline.add_stage(stage_2)
    
    with pytest.raises(ValueError, match="Undefined dependencies"):
        invalid_pipeline.get_execution_order()
    
@pytest.mark.skip(reason="Test not yet implemented.")
@pytest.mark.integration
def test_validate_dag_with_undefined_dependency(test_base_dir):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=test_base_dir)
    stage_1 = Stage(name="stage_1", config={})
    stage_2 = Stage(name="stage_2", config={})
    pipeline.add_stage(stage_1)
    pipeline.add_stage(stage_2)
    pipeline.dependencies["stage_2"] = ["stage_3"]  # Undefined stage_3
    
    with pytest.raises(ValueError, match="Stage 'stage_3' is not defined"):
        pipeline.validate_dag()

@pytest.mark.skip(reason="Test not yet implemented.")
@pytest.mark.integration
def test_validate_dag_with_self_dependency(test_base_dir):
    pipeline = Pipeline(name="test_pipeline", config={"ids": []}, directory=test_base_dir)
    stage_1 = Stage(name="stage_1", config={})
    pipeline.add_stage(stage_1)
    stage_1(data=stage_1.output["data"])  # Self-dependency

    with pytest.raises(ValueError, match="Stage 'stage_1' cannot depend on itself"):
        pipeline.validate_dag()
