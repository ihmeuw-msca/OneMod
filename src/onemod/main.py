"""Just something to get my example working."""

from __future__ import annotations

import fire
import json
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmodulename
from pathlib import Path

from jobmon.client.api import Tool
from jobmon.client.task_template import TaskTemplate

import onemod.stage as onemod_stages
from onemod.pipeline import Pipeline
from onemod.stage import Stage


def load_pipeline(filepath: str) -> Pipeline:
    return Pipeline.from_json(filepath)


def load_stage(
    filepath: Path | str,
    stage_name: str | None = None,
    from_pipeline: bool = False,
) -> Stage:
    """Load stage instance from JSON file.

    Parameters
    ----------
    filepath : Path or str
        Path to config file.
    stage_name : str or None, optional
        Stage name, required if `from_pipeline` is True.
        Default is None.
    from_pipeline : bool, optional
        Whether `filepath` is a pipeline or stage config file.
        Default is False.

    Returns
    -------
    Stage
        Stage instance.

    """
    stage_class = get_stage(filepath, stage_name, from_pipeline)
    return stage_class.from_json(filepath, stage_name, from_pipeline)


def get_stage(
    filepath: Path | str,
    stage_name: str | None = None,
    from_pipeline: bool = False,
) -> Stage:
    """Get stage class from JSON file.

    Parameters
    ----------
    filepath : Path or str
        Path to config file.
    stage_name : str or None, optional
        Stage name, required if `from_pipeline` is True.
        Default is None.
    from_pipeline : bool, optional
        Whether `filepath` is a pipeline or stage config file.
        Default is False.

    Returns
    -------
    Stage
        Stage class.

    """
    with open(filepath, "r") as f:
        config = json.load(f)
    if from_pipeline:
        try:
            config = config["stages"][stage_name]
        except KeyError:
            raise AttributeError(
                f"{config.name} does not have a '{stage_name}' stage"
            )
    try:
        if hasattr(onemod_stages, stage_type := config["type"]):
            return getattr(onemod_stages, stage_type)
    except KeyError:
        raise KeyError(
            "Stage config missing field 'type'; is this a pipeline config file?"
        )
    module_path = Path(config["module"])  # custom stage
    spec = spec_from_file_location(getmodulename(module_path), module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, stage_type)


def get_task_template(
    tool: Tool,
    stage_name: str,
    method: str,
    subsets: bool = False,
    params: bool = False,
) -> TaskTemplate:
    command_template = (
        "{python}"
        f" {__file__}"
        " --filepath {filepath}"
        " --stage_name stage_name"
        " --method method"
        " --from_pipeline"
    )

    node_args = []
    if subsets and method != "collect":
        command_template += " --subset_id {subset_id}"
        node_args.append("subset_id")
    if params and method != "collect":
        command_template += " --param_id {param_id}"
        node_args.append("param_id")

    return tool.get_task_template(
        template_name=f"{stage_name}_{method}_template",
        command_template=command_template,
        node_args=node_args,
        task_args=["filepath"],
        op_args=["python"],
    )


def evaluate(
    filepath: str,
    stage_name: str | None = None,
    from_pipeline: bool = False,
    method: str = "run",
    *args,
    **kwargs,
) -> None:
    if stage_name is None:
        Pipeline.evaluate(filepath, method, *args, **kwargs)
    else:
        stage_class = get_stage(filepath, stage_name, from_pipeline)
        stage_class.evaluate(filepath, stage_name, from_pipeline, method)


def main():
    fire.Fire(evaluate)
