import pytest

from onemod.redesign.pipeline import Pipeline
from onemod.redesign.stage import Stage


def test_pipeline_initialization(tmp_path):
    pipeline_dir = tmp_path / "test_pipeline"
    pipeline = Pipeline(name="test_pipeline", config={}, directory=pipeline_dir)
    assert pipeline.name == "test_pipeline"
    assert pipeline._stages == {}
    assert pipeline._dependencies == {}
    
def test_add_stage_without_dependencies(tmp_path):
    pipeline_dir = tmp_path / "test_pipeline"
    pipeline = Pipeline(name="test_pipeline", config={}, directory=pipeline_dir)
    stage = Stage(name="stage_a", config={})
    pipeline.add_stage(stage)
    assert "stage_a" in pipeline._stages
    assert pipeline._dependencies["stage_a"] == []

def test_add_stages_with_dependencies(tmp_path):
    pipeline_dir = tmp_path / "test_pipeline"
    pipeline = Pipeline(name="test_pipeline", config={}, directory=pipeline_dir)
    stage_a = Stage(name="stage_a", config={})
    stage_b = Stage(name="stage_b", config={})
    pipeline.add_stage(stage_a)
    pipeline.add_stage(stage_b, dependencies=["stage_a"])
    assert "stage_a" in pipeline._stages
    assert "stage_b" in pipeline._stages
    assert pipeline._dependencies["stage_b"] == ["stage_a"]

def test_duplicate_stage_error(tmp_path):
    pipeline_dir = tmp_path / "duplicate_stage_pipeline"
    pipeline = Pipeline(name="duplicate_stage_pipeline", config={}, directory=pipeline_dir)
    stage_a = Stage(name="stage_a", config={})
    pipeline.add_stage(stage_a)
    with pytest.raises(ValueError, match="stage 'stage_a' already exists"):
        pipeline.add_stage(stage_a)  # Add the same stage again

def test_undefined_dependency_error(tmp_path):
    pipeline_dir = tmp_path / "undefined_error_pipeline"
    pipeline = Pipeline(name="undefined_error_pipeline", config={}, directory=pipeline_dir)
    stage_b = Stage(name="stage_b", config={})
    with pytest.raises(ValueError, match="Dependency 'stage_a' not found in pipeline."):
        pipeline.add_stage(stage_b, dependencies=["stage_a"])

def test_pipeline_with_no_stages(tmp_path):
    pipeline_dir = tmp_path / "empty_pipeline"
    pipeline = Pipeline(name="empty_pipeline", config={}, directory=pipeline_dir)
    execution_order = pipeline.get_execution_order()
    assert execution_order == []

def test_pipeline_with_single_stage(tmp_path):
    pipeline_dir = tmp_path / "single_stage_pipeline"
    pipeline = Pipeline(name="single_stage_pipeline", config={}, directory=pipeline_dir)
    stage = Stage(name="stage_a", config={})
    pipeline.add_stage(stage)
    execution_order = pipeline.get_execution_order()
    assert execution_order == ["stage_a"]

def test_pipeline_with_valid_dependencies(tmp_path):
    pipeline_dir = tmp_path / "valid_pipeline"
    pipeline = Pipeline(name="valid_pipeline", config={}, directory=pipeline_dir)
    stage_a = Stage(name="stage_a", config={})
    stage_b = Stage(name="stage_b", config={})
    stage_c = Stage(name="stage_c", config={})
    pipeline.add_stage(stage_a)
    pipeline.add_stage(stage_b, dependencies=["stage_a"])
    pipeline.add_stage(stage_c, dependencies=["stage_b"])
    execution_order = pipeline.get_execution_order()
    assert execution_order == ["stage_a", "stage_b", "stage_c"]

def test_pipeline_with_cyclic_dependencies(tmp_path):
    pipeline_dir = tmp_path / "cyclic_pipeline"
    pipeline = Pipeline(name="cyclic_pipeline", config={}, directory=pipeline_dir)
    stage_a = Stage(name="stage_a", config={})
    stage_b = Stage(name="stage_b", config={})
    stage_c = Stage(name="stage_c", config={})
    pipeline.add_stage(stage_a, dependencies=["stage_c"])
    pipeline.add_stage(stage_b, dependencies=["stage_a"])
    pipeline.add_stage(stage_c, dependencies=["stage_b"])
    
    with pytest.raises(ValueError, match="Cycle detected"):
        pipeline.get_execution_order()

def test_pipeline_from_json_with_undefined_dependencies(tmp_path):
    pipeline_dir = tmp_path / "invalid_pipeline"
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

def test_validate():
    """
    Test the validate method of the Pipeline class.
    """
    pass

def test_compile():
    """
    Test the compile method of the Pipeline class.
    """
    pass
