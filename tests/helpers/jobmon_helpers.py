"""Dummy pipeline and stages to test Jobmon orchestration.

Stages
------
* simple_stage_1
  * input: pipeline_data
* simple_stage_fit_2
  * input: simple_stage_1
  * skips: predict
* simple_stage_predict_3
  * input: simple_stage_1
  * skips: fit
* parallel_stage_4
  * input: pipeline_data, simple_stage_1
  * groupby: sex_id
  * crossby: param
* parallel_stage_fit_5
  * input: simple_stage_1, parallel_stage_4
  * skips: predict
  * groupby: sex_id
* parallel_stage_predict_6
  * input: simple_stage_1, parallel_stage_4
  * skips: fit
  * crossby: param

Data Columns
------------
* pipeline_data: sex_id, year_id
* simple stage output: sex_id, year_id, input, stage
* parallel stage output: sex_id, year_id, input1, input2, stage
  * possible values for stage:
    * {stage}_subset_{subset_id}
    * {stage}_param_{param_id}
    * {stage}_subset_{subset_id}_param_{param_id}

"""

from pathlib import Path

import pandas as pd
from pydantic import Field

from onemod import Pipeline
from onemod.config import ModelConfig
from onemod.stage import ModelStage, Stage


class SimpleStage(Stage):
    _required_input: set[str] = {"input.csv"}
    _output: set[str] = {"output.csv"}
    _log: list[str] = []

    def get_log(self) -> list[str]:
        return self._log

    def run(self) -> None:
        self._log.append(f"run: name={self.name}")
        self._create_output()

    def fit(self) -> None:
        self._log.append(f"fit: name={self.name}")
        self._create_output()

    def predict(self) -> None:
        self._log.append(f"predict: name={self.name}")
        self._create_output()

    def _create_output(self) -> None:
        data = self.dataif.load(key="input", return_type="pandas_dataframe")
        data["input"] = data["stage"]
        data["stage"] = self.name
        self.dataif.dump(data, "output.csv", key="output")


class SimpleStageFit(SimpleStage):
    _skip: set[str] = {"predict"}


class SimpleStagePredict(SimpleStage):
    _skip: set[str] = {"fit"}


class ParallelConfig(ModelConfig):
    _crossable_params: set[str] = {"param"}

    param: int | set[int]


class ParallelStage(ModelStage):
    config: ParallelConfig
    _required_input: set[str] = {"input1.csv", "input2.csv"}
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
        self._create_output(subset_id, param_id)

    def fit(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        self._log.append(
            f"fit: name={self.name}, subset={subset_id}, param={param_id}"
        )
        self._create_output(subset_id, param_id)

    def predict(
        self, subset_id: int | None = None, param_id: int | None = None
    ) -> None:
        self._log.append(
            f"predict: name={self.name}, subset={subset_id}, param={param_id}"
        )
        self._create_output(subset_id, param_id)

    def _create_output(
        self, subset_id: int | None, param_id: int | None
    ) -> None:
        id_columns = list(self.config["id_columns"])
        if subset_id is None:
            left = self.dataif.load(
                key="input1", return_type="pandas_dataframe"
            )
            right = self.dataif.load(
                key="input2", return_type="pandas_dataframe"
            )
        else:
            left = self.get_stage_subset(
                subset_id=subset_id, key="input1"
            ).to_pandas()
            right = self.get_stage_subset(
                subset_id=subset_id, key="input2"
            ).to_pandas()

        data = pd.merge(
            left=left[id_columns + ["stage"]].rename(
                columns={"stage": "input1"}
            ),
            right=right[id_columns + ["stage"]].rename(
                columns={"stage": "input2"}
            ),
            on=id_columns,
        )

        stage_name = self.name
        if subset_id is not None:
            stage_name += f"_subset_{subset_id}"
        if param_id is not None:
            stage_name += f"_param_{param_id}"
        data["stage"] = stage_name

        self.dataif.dump(
            data, self._get_output_path(subset_id, param_id), key="output"
        )

    def collect(self) -> None:
        self._log.append(f"collect: name={self.name}")
        output = []
        for subset_id in self.subset_ids or [None]:
            for param_id in self.param_ids or [None]:
                output_dir = self.dataif.get_path(key="output")
                output_path = self._get_output_path(subset_id, param_id)
                if (output_dir / output_path).exists():
                    output.append(
                        self.dataif.load(
                            output_path,
                            key="output",
                            return_type="pandas_dataframe",
                        )
                    )
        if output:
            self.dataif.dump(pd.concat(output), "output.csv", key="output")

    def _get_output_path(
        self, subset_id: int | None, param_id: int | None
    ) -> str:
        if subset_id is None:
            return f"submodels/{param_id}/output.csv"
        if param_id is None:
            return f"submodels/{subset_id}/output.csv"
        return f"submodels/{subset_id}_{param_id}/output.csv"


class ParallelStageFit(ParallelStage):
    _skip: set[str] = {"predict"}
    _collect_after: set[str] = {"run", "fit"}


class ParallelStagePredict(ParallelStage):
    _skip: set[str] = {"fit"}
    _collect_after: set[str] = {"run", "predict"}


def create_data(directory: Path):
    data = pd.DataFrame(
        [[sex_id, year_id] for sex_id in [1, 2] for year_id in [1, 2]],
        columns=["sex_id", "year_id"],
    )
    data["stage"] = "pipeline_input"
    data.to_csv(directory / "data.csv", index=False)


def create_pipeline(directory: Path) -> Pipeline:
    return Pipeline(
        name="jobmon_test_pipeline",
        config={"id_columns": ["sex_id", "year_id"], "model_type": "gaussian"},
        directory=directory,
        data=directory / "data.csv",
    )


def create_stages() -> list[Stage]:
    return [
        SimpleStage(name="simple_stage_1", config={}),
        SimpleStageFit(name="simple_stage_fit_2", config={}),
        SimpleStagePredict(name="simple_stage_pred_3", config={}),
        ParallelStage(
            name="parallel_stage_4",
            config={"param": [1, 2]},
            groupby=["sex_id"],
        ),
        ParallelStageFit(
            name="parallel_stage_fit_5", config={"param": 1}, groupby=["sex_id"]
        ),
        ParallelStagePredict(
            name="parallel_stage_pred_6", config={"param": [1, 2]}
        ),
    ]


def define_dataflow(stages: list[Stage], input: Path) -> None:
    stage_1_output = stages[0](input=input)["output"]
    stage_2_output = stages[1](input=stage_1_output)["output"]
    stage_3_output = stages[2](input=stage_1_output)["output"]
    stage_4_output = stages[3](input1=input, input2=stage_1_output)["output"]
    stages[4](input1=stage_2_output, input2=stage_4_output)
    stages[5](input1=stage_3_output, input2=stage_4_output)


def setup_pipeline(directory: Path):
    create_data(directory)
    pipeline = create_pipeline(directory)
    stages = create_stages()
    pipeline.add_stages(stages)
    define_dataflow(stages, pipeline.data)
    pipeline.build()
    return pipeline
