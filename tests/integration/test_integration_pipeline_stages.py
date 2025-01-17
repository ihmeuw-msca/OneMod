import pytest

from onemod.config import StageConfig
from onemod.pipeline import Pipeline
from onemod.stage import Stage


class DummyStage(Stage):
    config: StageConfig
    _required_input: list[str] = ["data.parquet"]
    _optional_input: list[str] = ["priors.pkl"]
    _output: list[str] = ["predictions.parquet", "model.pkl"]

    def run(self) -> None:
        pass


@pytest.fixture
def stage_1(test_base_dir):
    stage_1 = DummyStage(name="stage_1", config={})
    stage_1(
        data=test_base_dir / "stage_1" / "data.parquet",
        covariates=test_base_dir / "stage_1" / "covariates.csv",
    )
    return stage_1


@pytest.fixture
def stage_2(stage_1):
    stage_2 = DummyStage(name="stage_2", config={})
    stage_2(
        data=stage_1.output["predictions"], covariates="/path/to/covariates.csv"
    )
    return stage_2


@pytest.mark.unit
def test_pipeline_initialization(test_base_dir):
    pipeline = Pipeline(
        name="test_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=test_base_dir,
    )
    assert pipeline.name == "test_pipeline"
    assert pipeline.stages == {}
    assert pipeline.dependencies == {}


@pytest.mark.integration
def test_add_stage_without_dependencies(test_base_dir):
    pipeline = Pipeline(
        name="test_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=test_base_dir,
    )
    stage = DummyStage(name="stage_1", config={})
    pipeline.add_stage(stage)
    assert "stage_1" in pipeline.stages
    assert pipeline.dependencies["stage_1"] == []


@pytest.mark.integration
def test_add_stages_with_dependencies(test_base_dir, stage_1, stage_2):
    pipeline = Pipeline(
        name="test_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=test_base_dir,
    )
    pipeline.add_stage(stage_1)
    pipeline.add_stage(stage_2)

    assert "stage_1" in pipeline.stages
    assert "stage_2" in pipeline.stages
    assert pipeline.dependencies["stage_2"] == ["stage_1"]


@pytest.mark.integration
def test_duplicate_stage_error(test_base_dir):
    pipeline = Pipeline(
        name="duplicate_stage_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=test_base_dir,
    )
    stage_1 = DummyStage(name="stage_1", config={})
    pipeline.add_stage(stage_1)
    with pytest.raises(ValueError, match="Stage 'stage_1' already exists"):
        pipeline.add_stage(stage_1)  # Add the same stage again


@pytest.mark.integration
def test_pipeline_with_no_stages(test_base_dir):
    pipeline = Pipeline(
        name="empty_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=test_base_dir,
    )
    execution_order = pipeline.get_execution_order()
    assert execution_order == []


@pytest.mark.integration
def test_pipeline_with_single_stage(test_base_dir):
    pipeline = Pipeline(
        name="single_stage_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=test_base_dir,
    )
    stage = DummyStage(name="stage_1", config={})
    pipeline.add_stage(stage)
    execution_order = pipeline.get_execution_order()
    assert execution_order == ["stage_1"]


@pytest.mark.integration
def test_pipeline_with_valid_dependencies(test_base_dir):
    pipeline = Pipeline(
        name="valid_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=test_base_dir,
    )
    stage_1 = DummyStage(name="stage_1", config={})
    stage_2 = DummyStage(name="stage_2", config={})
    stage_3 = DummyStage(name="stage_3", config={})
    pipeline.add_stage(stage_1)
    pipeline.add_stage(stage_2)
    pipeline.add_stage(stage_3)

    stage_1(data=test_base_dir / "stage_1" / "data.parquet")
    stage_2(data=stage_1.output["predictions"])
    stage_3(
        data=stage_1.output["predictions"],
        selected_covs=stage_2.output["predictions"],
    )

    execution_order = pipeline.get_execution_order()
    assert execution_order == ["stage_1", "stage_2", "stage_3"]


@pytest.mark.integration
def test_pipeline_with_cyclic_dependencies(test_base_dir):
    pipeline = Pipeline(
        name="cyclic_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=test_base_dir,
    )
    stage_1 = DummyStage(name="stage_1", config={})
    stage_2 = DummyStage(name="stage_2", config={})
    stage_3 = DummyStage(name="stage_3", config={})
    pipeline.add_stage(stage_1)
    pipeline.add_stage(stage_2)
    pipeline.add_stage(stage_3)

    stage_1(data=test_base_dir / "stage_1" / "data.parquet")
    stage_2(data=stage_1.output["predictions"])
    stage_1(
        data=stage_2.output["predictions"]
    )  # Redefined with cyclic dependency

    with pytest.raises(ValueError, match="Cycle detected"):
        pipeline.get_execution_order()


# FIXME: Since output items no longer use self.directory, no error is thrown.
# However, there should be an error thrown if the stage hasn't been added to
# the pipeline. We need to add a check in __call__ to make sure the stage has
# been added to the pipeline.
@pytest.mark.skip(reason="Pending implementation")
@pytest.mark.integration
def test_pipeline_with_undefined_dependencies(test_base_dir):
    pipeline_dir = test_base_dir / "invalid_pipeline"
    pipeline = Pipeline(
        name="invalid_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=pipeline_dir,
    )

    stage_1 = DummyStage(name="stage_1", config={})
    stage_2 = DummyStage(name="stage_2", config={})
    pipeline.add_stage(stage_2)  # stage_1 never added

    stage_1(data=test_base_dir / "stage_1" / "data.parquet")

    with pytest.raises(
        AttributeError, match="Stage 'stage_1' directory has not been set"
    ):
        stage_2(data=stage_1.output["predictions"])


@pytest.mark.integration
def test_pipeline_with_duplicate_groupby(test_base_dir):
    pipeline_dir = test_base_dir / "duplicate_groupby_pipeline"
    pipeline = Pipeline(
        name="duplicate_groupby_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=pipeline_dir,
        groupby=["age_group_id", "age_group_id", "location_id"],
        groupby_data="path/to/data",
    )

    assert pipeline.groupby == ["age_group_id", "location_id"]


@pytest.mark.integration
def test_validate_dag_with_self_dependency(test_base_dir):
    pipeline = Pipeline(
        name="test_pipeline",
        config={"id_columns": [], "model_type": "binomial"},
        directory=test_base_dir,
    )
    stage_1 = DummyStage(name="stage_1", config={})

    pipeline.add_stage(stage_1)

    with pytest.raises(
        ValueError, match="Circular dependencies for stage_1 input"
    ):
        stage_1(data=stage_1.output["predictions"])  # Self-dependency
