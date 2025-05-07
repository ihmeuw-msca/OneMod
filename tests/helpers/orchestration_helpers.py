"""Dummy pipeline and stages to test pipeline orchestration."""

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from onemod import Pipeline, load_stage
from onemod.dtypes import Data
from onemod.stage import Stage


class SimpleStage(Stage):
    """Used to test skip and method-specific dependencies."""

    # TODO: Update once method-specific dependencies implemented

    _required_input: dict[str, Data] = {
        "fit_input": Data(methods=["run", "fit"], format="csv"),
        "predict_input": Data(methods=["run", "predict"], format="csv"),
    }
    _output_items: dict[str, Data] = {
        "fit_output": Data(methods=["run", "fit"], format="csv"),
        "predict_output": Data(methods=["run", "predict"], format="csv"),
    }
    _log: list[str] = []

    def get_log(self) -> list[str]:
        return self._log

    def _run(self, **kwargs) -> None:
        self._log.append(f"run: name={self.name}")
        self._create_output("fit", **kwargs)
        self._create_output("predict", **kwargs)

    def _fit(self) -> None:
        self._log.append(f"fit: name={self.name}")
        self._create_output("fit")

    def _predict(self) -> None:
        self._log.append(f"predict: name={self.name}")
        self._create_output("predict")

    def _create_output(self, method: str, **kwargs) -> None:
        data = self.dataif.load(key=f"{method}_input")
        data["input"] = data["stage"]
        data["stage"] = self.name
        data["method"] = method

        for key, value in kwargs.items():
            data[key] = value

        self.dataif.dump(data, f"{method}_output.csv", key="output")


class SimpleStageFit(SimpleStage):
    _skip: list[str] = ["predict"]
    _required_input: dict[str, Data] = {
        "fit_input": Data(methods=["run", "fit"], format="csv")
    }
    _output_items: dict[str, Data] = {
        "fit_output": Data(methods=["run", "fit"], format="csv")
    }

    def _run(self, **kwargs) -> None:
        self._log.append(f"run: name={self.name}")
        self._create_output("fit", **kwargs)


class SimpleStagePredict(SimpleStage):
    _skip: list[str] = ["fit"]
    _required_input: dict[str, Data] = {
        "predict_input": Data(methods=["run", "fit"], format="csv")
    }
    _output_items: dict[str, Data] = {
        "predict_output": Data(methods=["run", "fit"], format="csv")
    }

    def _run(self, **kwargs) -> None:
        self._log.append(f"run: name={self.name}")
        self._create_output("predict", **kwargs)


class ParallelStage(Stage):
    """Used to test groupby, crossby, and collect_after."""

    # TODO: Update once stage instance can be passed as input

    _required_input: dict[str, Data] = {
        "input1": Data(format="directory"),
        "input2": Data(format="directory"),
    }
    _output_items: dict[str, Data] = {"output": Data(format="csv")}
    _collect_after: list[str] = ["run", "fit", "predict"]
    _log: list[str] = []

    def get_log(self) -> list[str]:
        return self._log

    def _run(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        self._log.append(
            f"run: name={self.name}, subset={subset}, paramset={paramset}"
        )
        self._create_output("run", subset, paramset)

    def _fit(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        self._log.append(
            f"fit: name={self.name}, subset={subset}, paramset={paramset}"
        )
        self._create_output("fit", subset, paramset)

    def _predict(
        self,
        subset: dict[str, Any] | None = None,
        paramset: dict[str, Any] | None = None,
    ) -> None:
        self._log.append(
            f"predict: name={self.name}, subset={subset}, paramset={paramset}"
        )
        self._create_output("predict", subset, paramset)

    def collect(self) -> None:
        self._log.append(f"collect: name={self.name}")
        output = _collect_stage_output(self)
        self.dataif.dump(output, "output.csv", key="output")
        shutil.rmtree(
            Path(self.dataif.get_full_path("submodels", key="output"))
        )

    def _create_output(
        self,
        method: str,
        subset: dict[str, Any] | None,
        paramset: dict[str, Any] | None,
    ) -> None:
        data = pd.merge(
            left=self._load_input("input1", method, subset),
            right=self._load_input("input2", method, subset),
            on=list(self.config["ids"]),
        )

        stage_name = self.name
        if subset is not None:
            stage_name += "__subset_" + self._get_str(subset)
        if paramset is not None:
            stage_name += "__paramset_" + self._get_str(paramset)
        data["stage"] = stage_name
        data["method"] = method

        self.dataif.dump(
            data, self._get_output_path(method, subset, paramset), key="output"
        )

    def _load_input(
        self, input_name: str, method: str, subset: dict[str, Any] | None
    ) -> pd.DataFrame:
        # Load input data
        input_path = Path(self.dataif.get_path(input_name))
        if input_path == self.dataif.get_path("directory"):
            data = self.dataif.load("data.csv", key="directory", subset=subset)
        else:
            # input item is upstream stage output directory
            # collect upstream stage results if necessary
            upstream_stage = load_stage(
                config=self.dataif.get_path("config"),
                stage_name=input_path.name,
            )
            if method in upstream_stage.collect_after:
                data = self.dataif.load(
                    "output.csv", key=input_name, subset=subset
                )
            else:
                data = _collect_stage_output(upstream_stage, method)
                if subset is not None:
                    data = self.get_subset(data, subset)

        # Rename stage column as input1 or input1
        return data[self.config["ids"] + ["stage"]].rename(
            columns={"stage": input_name}
        )

    def _get_output_path(
        self,
        method: str,
        subset: dict[str, Any] | None,
        paramset: dict[str, Any] | None,
    ) -> str:
        output_dir = f"submodels/{method}/"
        if subset is None:
            if paramset is None:
                return output_dir + "output.csv"
            else:
                return output_dir + f"{self._get_str(paramset)}/output.csv"
        if paramset is None:
            return output_dir + f"{self._get_str(subset)}/output.csv"
        return (
            output_dir
            + f"{self._get_str(subset)}__{self._get_str(paramset)}/output.csv"
        )

    @staticmethod
    def _get_str(subset: dict[str, Any]) -> str:
        return "_".join(str(value) for value in subset.values())


class ParallelStageFit(ParallelStage):
    _collect_after: list[str] = ["run", "fit"]


class ParallelStagePredict(ParallelStage):
    _collect_after: list[str] = ["run", "predict"]


def _collect_stage_output(stage, method: str | None = None) -> pd.DataFrame:
    output_dir = Path(stage.dataif.get_path(key="output"))
    if method is None:
        method = _get_method_name(stage.name, output_dir)

    output = []
    for subset, paramset in stage.get_submodels():
        try:
            output.append(
                stage.dataif.load(
                    stage._get_output_path(method, subset, paramset),
                    key="output",
                )
            )
        except FileNotFoundError:
            pass

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
        [
            [sex_id, age_group_id, year_id]
            for sex_id in [1, 2]
            for age_group_id in [1, 2, 3]
            for year_id in [1, 2]
        ],
        columns=["sex_id", "age_group_id", "year_id"],
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
    pipeline = Pipeline(name="test_simple_pipeline", directory=directory)

    # Create stages and add to pipeline
    run_1 = SimpleStage(name="run_1")  # type: ignore
    fit_2 = SimpleStageFit(name="fit_2")  # type: ignore
    predict_3 = SimpleStagePredict(name="predict_3")  # type: ignore
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
    pipeline = Pipeline(
        name="test_parallel_pipeline",
        directory=directory,
        config={"ids": ["sex_id", "age_group_id", "year_id"]},  # type: ignore
        groupby_data=data_path,
    )

    # Create stages and add to pipeline
    run_1 = ParallelStage(
        name="run_1",
        config={"param1": [1, 2], "param2": [1, 2, 3]},  # type: ignore
        groupby=["sex_id", "age_group_id"],
        crossby=["param1", "param2"],
    )
    fit_2 = ParallelStageFit(
        name="fit_2",
        config={"param": 1},  # type: ignore
        groupby=["sex_id"],
    )
    predict_3 = ParallelStagePredict(
        name="predict_3",
        config={"param": [1, 2]},  # type: ignore
        crossby=["param"],
    )
    run_4 = ParallelStage(
        name="run_4",
        config={"param": [1, 2]},  # type: ignore
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


def assert_parallel_logs(
    stage: ParallelStage,
    method: str,
    subsets: dict | None = None,
    paramsets: dict | None = None,
    collect: bool | None = None,
) -> None:
    """Assert expected methods, subsets, and paramsets logged."""
    log = stage.get_log()
    for subset, paramset in stage.get_submodels(subsets, paramsets):
        assert (
            f"{method}: name={stage.name}, subset={subset}, paramset={paramset}"
            in log
        )
    if method in stage._collect_after:
        if collect is None:
            if subsets is None and paramsets is None:
                assert f"collect: name={stage.name}" in log
        elif collect:
            assert f"collect: name={stage.name}" in log


def assert_simple_output(stage: SimpleStage, method: str, **kwargs) -> None:
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
    if (upstream_stage := stage.input[f"{method}_input"].stage) is None:
        assert input_column[0] == "pipeline_input"
    else:
        assert input_column[0] == upstream_stage

    # Check method column
    assert len(method_column := output["method"].unique()) == 1
    assert method_column[0] == method


def assert_parallel_output(
    stage: ParallelStage,
    method: str,
    subsets: dict | None = None,
    paramsets: dict | None = None,
    collect: bool | None = None,
) -> None:
    """Assert expected columns in stage output."""
    # Load stage output
    if method in stage.collect_after:
        if collect is None:
            if subsets is None and paramsets is None:
                output = stage.dataif.load("output.csv", key="output")
            else:
                output = _collect_stage_output(stage, method)
        elif collect:
            output = stage.dataif.load("output.csv", key="output")
        else:
            output = _collect_stage_output(stage, method)
    else:
        output = _collect_stage_output(stage, method)

    # Check stage column
    for subset, paramset in stage.get_submodels(subsets, paramsets):
        expected_name = stage.name
        if subset is not None:
            expected_name += f"__subset_{stage._get_str(subset)}"
        if paramset is not None:
            expected_name += f"__paramset_{stage._get_str(paramset)}"
        assert expected_name in output["stage"].values

    # Check input columns
    for input_name in ["input1", "input2"]:
        if (upstream_stage := stage.input[input_name].stage) is None:
            assert len(input_column := output[input_name].unique()) == 1
            assert input_column[0] == "pipeline_input"
        else:
            for _, row in output.iterrows():
                row_stage = row["stage"]
                row_input = row[input_name]
                assert row_input.startswith(upstream_stage)
                if "subset" in row_stage and "subset" in row_input:
                    row_stage.split("__")[1] == row_input.split("__")[1]

    # Check method column
    assert len(method_column := output["method"].unique()) == 1
    assert method_column[0] == method
