"""Run pipeline or stage."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmodulename
from pathlib import Path
from typing import Literal

import fire
from pydantic import validate_call

import onemod.stage as onemod_stages
from onemod.pipeline import Pipeline
from onemod.stage import Stage


def init(directory: Path | str) -> None:
    """Initialize project directory."""
    raise NotImplementedError()


def load_pipeline(config: str) -> Pipeline:
    """Load pipeline instance from JSON file.

    Parameters
    ----------
    config : Path or str
        Path to config file.

    Returns
    -------
    Pipeline
        Pipeline instance.

    """
    return Pipeline.from_json(config)


def load_stage(config: Path | str, stage_name: str) -> Stage:
    """Load stage instance from JSON file.

    Parameters
    ----------
    config : Path or str
        Path to config file.
    stage_name : str
        Stage name.

    Returns
    -------
    Stage
        Stage instance.

    """
    stage_class = _get_stage(config, stage_name)
    return stage_class.from_json(config, stage_name)


def _get_stage(config: Path | str, stage_name: str) -> Stage:
    """Get stage class from JSON file.

    Parameters
    ----------
    config : Path or str
        Path to config file.
    stage_name : str
        Stage name.

    Returns
    -------
    Stage
        Stage class.

    """
    with open(config, "r") as f:
        config_dict = json.load(f)
    if stage_name not in config_dict["stages"]:
        raise KeyError(f"Config does not contain a stage named '{stage_name}'")
    config_dict = config_dict["stages"][stage_name]
    if hasattr(onemod_stages, stage_type := config_dict["type"]):
        return getattr(onemod_stages, stage_type)
    if "module" not in config_dict:
        raise KeyError(f"Config does not contain a module for {stage_name}")
    return _get_custom_stage(stage_type, config_dict["module"])


def _get_custom_stage(stage_type: str, module: str) -> Stage:
    """Get custom stage class from file.

    Parameters
    ----------
    stage_type : str
        Name of custom stage class.
    module : str
        Path to Python module containing custom stage class definition.

    Returns
    -------
    Stage
        Custom stage class.

    """
    module_path = Path(module)
    spec = spec_from_file_location(getmodulename(module_path), module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, stage_type)


@validate_call
def evaluate(
    config: Path | str,
    stage_name: str | None = None,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    backend: Literal["local", "jobmon"] = "local",
    **kwargs,
) -> None:
    """Evaluate pipeline or stage method.

    Parameters
    ----------
    config : Path or str
        Path to config file.
    stage_name : str or None, optional
        Name of stage to evaluate. If None, evaluate pipeline.
        Default is None.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    backend : str, optional
        Whether to evaluate the method locally or with Jobmon.
        Default is 'local'.

    Other Parameters
    ----------------
    subset_id : int, optional
        Submodel data subset ID. Only used for model stages.
    param_id : int, optional
        Submodel parameter set ID. Only used for model stages.
    cluster : str, optional
        Cluster name. Required if `backend` is 'jobmon'.
    resources : Path or str, optional
        Path to resources yaml file. Required if `backend` is 'jobmon'.

    """
    if stage_name is None:
        model = load_pipeline(config)
    else:
        model = load_stage(config, stage_name)
    model.evaluate(method, backend, **kwargs)


def call_function(method: str, **kwargs):
    if method == "init":
        init(**kwargs)
    elif method in ["run", "fit", "predict", "collect"]:
        evaluate(method=method, **kwargs)
    else:
        raise ValueError(f"Invalid function name: {method}")


def main():
    fire.Fire(call_function)


if __name__ == "__main__":
    main()
