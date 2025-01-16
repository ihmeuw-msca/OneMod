"""Dummy pipeline and stages to test pipeline orchestration."""

import shutil
from itertools import product
from pathlib import Path

import pandas as pd

from onemod import Pipeline, load_stage
from onemod.config import StageConfig
from onemod.dtypes import Data
from onemod.stage import Stage
from onemod.utils.subsets import get_subset


class SimpleStage(Stage):
    """Used to test skip and method-specific dependencies."""

    # TODO: Update once method-specific dependencies implemented

    config: StageConfig = StageConfig()
    _required_input: set[str] = {"fit_input.csv", "predict_input.csv"}
    _output: set[str] = {"fit_output.csv", "predict_output.csv"}
    _log: list[str] = []

    def get_log(self) -> list[str]:
        return self._log

    def run(self) -> None:
        self._log.append(f"run: name={self.name}")
        self._create_output("fit")
        self._create_output("predict")

    def fit(self) -> None:
        self._log.append(f"fit: name={self.name}")
        self._create_output("fit")

    def predict(self) -> None:
        self._log.append(f"predict: name={self.name}")
        self._create_output("predict")

    def _create_output(self, method: str) -> None:
        data = self.dataif.load(key=f"{method}_input")
        data["input"] = data["stage"]
        data["stage"] = self.name
        data["method"] = method
        self.dataif.dump(data, f"{method}_output.csv", key="output")


class SimpleStageFit(SimpleStage):
    _required_input: set[str] = {"fit_input.csv"}
    _output: set[str] = {"fit_output.csv"}
    _skip: set[str] = {"predict"}

    def run(self) -> None:
        self._log.append(f"run: name={self.name}")
        self._create_output("fit")


class SimpleStagePredict(SimpleStage):
    _required_input: set[str] = {"predict_input.csv"}
    _output: set[str] = {"predict_output.csv"}
    _skip: set[str] = {"fit"}

    def run(self) -> None:
        self._log.append(f"run: name={self.name}")
        self._create_output("predict")


class ParallelConfig(StageConfig):
    # TODO: Remove class once crossby changed
    # TODO: Change set to list once that gets sorted out
    _crossable_params: set[str] = {"param"}

    param: int | set[int]


class ParallelStage(Stage):
    """Used to test groupby, crossby, and collect_after."""

    # TODO: Update once stage instance can be passed as input

    config: ParallelConfig
    _required_input: set[str] = {"input1", "input2"}
    _output: set[str] = {"output.csv"}
    _collect_after: set[str] = {"run", "fit", "predict"}
    _log: list[str] = []

    def get_log(self) -> list[str]:
        return self._log

    def run(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        self._log.append(
            f"run: name={self.name}, subset={subset_id}, param={param_id}"
        )
        self._create_output("run", subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        self._log.append(
            f"fit: name={self.name}, subset={subset_id}, param={param_id}"
        )
        self._create_output("fit", subset_id, param_id)

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        self._log.append(
            f"predict: name={self.name}, subset={subset_id}, param={param_id}"
        )
        self._create_output("predict", subset_id, param_id)

    def collect(self, delete_submodel_results: bool = True) -> None:
        self._log.append(f"collect: name={self.name}")

        output = _collect_stage_output(self)
        self.dataif.dump(output, "output.csv", key="output")

        if delete_submodel_results:
            shutil.rmtree(
                Path(self.dataif.get_full_path("submodels", key="output"))
            )

    def _create_output(
        self, method: str, subset_id: int | None, param_id: int | None
    ) -> None:
        data = pd.merge(
            left=self._load_input("input1", method, subset_id),
            right=self._load_input("input2", method, subset_id),
            on=list(self.config["id_columns"]),
        )

        stage_name = self.name
        if subset_id is not None:
            stage_name += f"__subset_{subset_id}"
        if param_id is not None:
            stage_name += f"__param_{param_id}"
        data["stage"] = stage_name
        data["method"] = method

        self.dataif.dump(
            data,
            self._get_output_path(method, subset_id, param_id),
            key="output",
        )

    def _load_input(
        self, input_name: str, method: str, subset_id: int | None
    ) -> pd.DataFrame:
        # Load input data
        input_path = Path(self.dataif.get_path(input_name))
        if input_path == self.dataif.get_path("directory"):
            data = self.dataif.load("data.csv", key="directory")
        else:
            # input item is upstream stage output directory
            # collect upstream stage results if necessary
            upstream_stage = load_stage(
                config=self.dataif.get_path("config"),
                stage_name=input_path.name,
            )
            if method in upstream_stage.collect_after:
                data = self.dataif.load("output.csv", key=input_name)
            else:
                data = _collect_stage_output(upstream_stage, method)

        # Filter by subset_id if necessary
        if subset_id is not None:
            data = get_subset(
                data, self.dataif.load("subsets.csv", key="output"), subset_id
            )

        # Rename stage column as input1 or input1
        return data[list(self.config["id_columns"]) + ["stage"]].rename(
            columns={"stage": input_name}
        )

    def _get_output_path(
        self, method: str, subset_id: int | None, param_id: int | None
    ) -> str:
        if subset_id is None:
            return f"submodels/{method}/{param_id}/output.csv"
        if param_id is None:
            return f"submodels/{method}/{subset_id}/output.csv"
        return f"submodels/{method}/{subset_id}_{param_id}/output.csv"


class ParallelStageFit(ParallelStage):
    _collect_after: set[str] = {"run", "fit"}


class ParallelStagePredict(ParallelStage):
    _collect_after: set[str] = {"run", "predict"}


def _collect_stage_output(stage, method: str | None = None) -> pd.DataFrame:
    output_dir = Path(stage.dataif.get_path(key="output"))
    if method is None:
        method = _get_method_name(stage.name, output_dir)

    output = []
    for subset_id in stage.subset_ids or [None]:
        for param_id in stage.param_ids or [None]:
            output.append(
                stage.dataif.load(
                    stage._get_output_path(method, subset_id, param_id),
                    key="output",
                )
            )

    return pd.concat(output)


def _get_method_name(stage_name: str, output_dir: Path) -> str:
    """Get which method was called based on submodel directories."""
    sub_dirs = [
        item for item in (output_dir / "submodels").iterdir() if item.is_dir()
    ]
    method_dirs = [
        item.name for item in sub_dirs if item.name in ["run", "fit", "predict"]
    ]
    if len(method_dirs) == 0:
        raise ValueError(
            f"No valid submodel directory exists for stage '{stage_name}'"
        )
    if len(method_dirs) > 1:
        raise ValueError(
            f"Two submodel directories exist for stage '{stage_name}': "
            f"{method_dirs}, "
            "please delete previous results"
        )
    return method_dirs[0]


def create_data(directory: Path) -> Path:
    data = pd.DataFrame(
        [[sex_id, year_id] for sex_id in [1, 2] for year_id in [1, 2]],
        columns=["sex_id", "year_id"],
    )
    data["stage"] = "pipeline_input"

    if not directory.exists():
        directory.mkdir(parents=True)
    data.to_csv(data_path := directory / "data.csv", index=False)

    return data_path


def setup_simple_pipeline(directory: Path) -> Pipeline:
    # Create input data
    data_path = create_data(directory)

    # Create pipeline
    pipeline = Pipeline(
        name="test_simple_pipeline", config={}, directory=directory
    )

    # Create stages and add to pipeline
    run_1 = SimpleStage(name="run_1")
    fit_2 = SimpleStageFit(name="fit_2")
    predict_3 = SimpleStagePredict(name="predict_3")
    pipeline.add_stages([run_1, fit_2, predict_3])

    # Define dataflow
    run_1(fit_input=data_path, predict_input=data_path)
    fit_2(fit_input=run_1.output["fit_output"])
    predict_3(predict_input=run_1.output["predict_output"])

    # TODO: Uncomment once method-specific dependencies implemented
    # run_4 = SimpleStage(name="run_4")
    # run_5 = SimpleStage(name="run_5")
    # pipeline.add_stages([run_4, run_5])
    # run_4(
    #     fit_input=fit_2.output["fit_output"],
    #     predict_input=predict_3.output["predict_output"],
    # )
    # run_5(
    #     fit_input=predict_3.output["predict_output"],
    #     predict_input=fit_2.output["fit_output"],
    # )

    # Build and return pipeline
    pipeline.build()
    return pipeline


def setup_parallel_pipeline(directory: Path) -> Pipeline:
    # Create input data
    data_path = create_data(directory)

    # Create pipeline
    # Note: This doesn't test passing of pipeline groupby to stages
    pipeline = Pipeline(
        name="test_parallel_pipeline",
        config={"id_columns": ["sex_id", "year_id"]},
        directory=directory,
        groupby_data=data_path,
    )

    # Create stages and add to pipeline
    # TODO: Add crossby=param once crossby changed
    run_1 = ParallelStage(
        name="run_1",
        config={"param": [1, 2]},
        groupby=["sex_id"],
        crossby=["param"],
    )
    fit_2 = ParallelStageFit(
        name="fit_2", config={"param": 1}, groupby=["sex_id"]
    )
    predict_3 = ParallelStagePredict(
        name="predict_3", config={"param": [1, 2]}, crossby=["param"]
    )
    run_4 = ParallelStage(
        name="run_4",
        config={"param": [1, 2]},
        groupby=["sex_id"],
        crossby=["param"],
    )
    pipeline.add_stages([run_1, fit_2, predict_3, run_4])

    # Define dataflow
    # TODO: Update once stage instance can be passed as input
    run_1(input1=directory, input2=directory)
    fit_2(
        input1=Data(
            stage="run_1", path=directory / "run_1", format="directory"
        ),
        input2=directory,
    )
    predict_3(
        input1=Data(
            stage="run_1", path=directory / "run_1", format="directory"
        ),
        input2=directory,
    )
    run_4(
        input1=Data(
            stage="fit_2", path=directory / "fit_2", format="directory"
        ),
        input2=Data(
            stage="predict_3", path=directory / "predict_3", format="directory"
        ),
    )

    # Build and return pipeline
    pipeline.build()
    return pipeline


def assert_simple_logs(stage: SimpleStage, method: str) -> None:
    """Assert expected methods are logged."""
    if method in stage.skip:
        return

    log = stage.get_log()
    assert f"{method}: name={stage.name}" in log


def assert_parallel_logs(stage: ParallelStage, method: str) -> None:
    """Assert expected methods, subset_ids, and param_ids logged."""
    log = stage.get_log()
    for subset_id, param_id in product(
        stage.subset_ids or [None], stage.param_ids or [None]
    ):
        assert (
            f"{method}: name={stage.name}, subset={subset_id}, param={param_id}"
            in log
        )
    if method in stage._collect_after:
        assert f"collect: name={stage.name}" in log


def assert_simple_output(stage: SimpleStage, method: str) -> None:
    """Assert expected columns are in stage output."""
    if method in stage.skip:
        return

    if method == "run":
        for method in {"fit", "predict"} - set(stage.skip):
            _assert_simple_output(stage, method)
    else:
        _assert_simple_output(stage, method)


def _assert_simple_output(stage: SimpleStage, method: str) -> None:
    # Load stage output
    output = stage.dataif.load(f"{method}_output.csv", key="output")

    # Check stage column
    assert len(stage_column := output["stage"].unique()) == 1
    assert stage_column[0] == stage.name

    # Check input columns
    assert len(input_column := output["input"].unique()) == 1
    if isinstance(stage_input := stage.input[f"{method}_input"], Path):
        assert input_column[0] == "pipeline_input"
    else:
        assert input_column[0] == stage_input.stage

    # Check method column
    assert len(method_column := output["method"].unique()) == 1
    assert method_column[0] == method


def assert_parallel_output(stage: ParallelStage, method: str) -> None:
    """Assert expected columns in stage output."""
    # Load stage output
    if method in stage.collect_after:
        output = stage.dataif.load("output.csv", key="output")
    else:
        output = _collect_stage_output(stage, method)

    # Check stage column
    for subset_id in stage.subset_ids or [None]:
        for param_id in stage.param_ids or [None]:
            expected_name = stage.name
            if subset_id is not None:
                expected_name += f"__subset_{subset_id}"
            if param_id is not None:
                expected_name += f"__param_{param_id}"
            assert expected_name in output["stage"].values

    # Check input columns
    for input_name in ["input1", "input2"]:
        input_column = output[input_name].unique()
        input_item = stage.input[input_name]

        if isinstance(input_item, Path):
            assert len(input_column) == 1
            assert input_column[0] == "pipeline_input"
        else:
            upstream_stage = input_item.stage
            for _, row in output.iterrows():
                row_stage = row["stage"]
                row_input = row[input_name]
                assert row_input.startswith(upstream_stage)
                if "subset" in row_stage and "subset" in row_input:
                    row_stage.split("__")[1] == row_input.split("__")[1]

    # Check method column
    assert len(method_column := output["method"].unique()) == 1
    assert method_column[0] == method
