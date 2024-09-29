"""Pipeline class."""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pydantic import BaseModel, computed_field

from onemod.config import PipelineConfig
from onemod.stage import CrossedStage, GroupedStage, Stage


class Pipeline(BaseModel):
    """
    Notes
    -----
    * `data` is raw input data used to figure out grouped stage subsets
       when initializing the pipeline; stages will get their data from
       other stages (with the exception of any preprocessing stages)
    * `data` is only required if using grouped stages
    * TODO: Define dependencies between stages
    * TODO: DAG creation, validation
    * TODO: Run locally or on Jobmon

    """

    name: str
    config: PipelineConfig
    directory: Path
    data: Path | None = None
    groupby: set[str] = set()
    _stages: dict[str, Stage] = {}  # set by Pipeline.add_stage

    @computed_field
    @property
    def stages(self) -> dict[str, Stage]:
        return self._stages

    def model_post_init(self, *args, **kwargs) -> None:
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

    @classmethod
    def from_json(cls, filepath: Path | str) -> Pipeline:
        """Load pipeline from JSON file.

        Parameters
        ----------
        filepath : Path or str
            Path to config file.

        Returns
        -------
        Pipeline
            Pipeline instance.

        """
        with open(filepath, "r") as f:
            config = json.load(f)
        stages = config.pop("stages", None)
        pipeline = cls(**config)
        if stages is not None:
            pipeline.add_stages(stages)
        return pipeline

    def to_json(self, filepath: Path | str | None = None) -> None:
        """Save pipeline as JSON file.

        Parameters
        ----------
        filepath : Path, str, or None, optional
            Where to save config file. If None, file is saved at
            pipeline.directory / (pipeline.name + ".json").
            Default is None.

        """
        filepath = filepath or self.directory / (self.name + ".json")
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4, serialize_as_any=True))

    def add_stages(
        self, stages: list[Stage | str], filepath: Path | str | None = None
    ) -> None:
        """Add stages to pipeline."""
        for stage in stages:
            self.add_stage(stage, filepath)

    def add_stage(
        self, stage: Stage | str, filepath: Path | str | None = None
    ) -> None:
        """Add stage to pipeline."""
        if isinstance(stage, str):
            stage = self._stage_from_json(stage, filepath)
        self._add_stage(stage)

    def _add_stage(self, stage: Stage) -> None:
        """Add stage object to pipeline."""
        if stage.name in self.stages:
            raise ValueError(f"stage '{stage.name}' already exists")
        stage.config.update(self.config)
        stage.directory = self.directory / stage.name
        if isinstance(stage, GroupedStage):
            if self.data is None:
                raise ValueError("data is required for GroupedStage")
            stage.groupby.update(self.groupby)
            stage.create_stage_subsets(self.data)
        if isinstance(stage, CrossedStage):
            if stage.config.crossable_params:
                stage.create_stage_params()
        self._stages[stage.name] = stage

    def _stage_from_json(self, stage: str, filepath: Path | str) -> Stage:
        """Load stage object from JSON.

        Notes
        -----
        * There may be an easier way to load built-in stages

        """
        if stage in self.stages:
            raise ValueError(f"stage '{stage}' already exists")
        with open(filepath, "r") as f:
            stage_json = json.load(f)
        spec = spec_from_file_location(stage_json["module"])
        module = module_from_spec(spec)
        spec.loader.exec_module(module_from_spec(spec))
        stage_class = module.__getattribute__(stage_json["type"])
        return stage_class(**stage_json)

    def run(self) -> None:
        """Run pipeline.

        Notes
        -----
        * These functions will handle workflow creation
        * Maybe allow subsets of the DAG to be run (e.g., predict for a
          single location)

        """
        raise NotImplementedError()

    def resume(self) -> None:
        """Resume pipeline."""
        raise NotImplementedError()

    def fit(self) -> None:
        """Fit pipeline model.

        Notes
        -----
        * Skip any stage that has 'fit' in `skip_if` attribute

        """
        raise NotImplementedError()

    def predict(self) -> None:
        """Predict pipeline model.

        Notes
        -----
        * Skip any stage that has 'predict' in `skip_if` attribute

        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
