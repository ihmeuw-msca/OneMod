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


def load_pipeline(config: Path | str) -> Pipeline:
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
    stage = stage_class.from_json(config, stage_name)
    return stage


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
    stage_type = config_dict["type"]

    if "module" in config_dict:
        return _get_custom_stage(stage_type, config_dict["module"])
    if hasattr(onemod_stages, stage_type):
        return getattr(onemod_stages, stage_type)
    raise KeyError(f"Config does not contain a module for {stage_name}")


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

    module_name = getmodulename(module_path)
    if module_name is None:
        raise ValueError(f"Could not determine module name from {module_path}")

    spec = spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_path}")

    if spec.loader is None:
        raise ImportError(f"Module spec for {module_path} has no loader")

    loaded_module = module_from_spec(spec)
    spec.loader.exec_module(loaded_module)

    return getattr(loaded_module, stage_type)


@validate_call
def evaluate(
    config: Path | str,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    stages: str | set[str] | None = None,
    backend: Literal["local", "jobmon"] = "local",
    **kwargs,
) -> None:
    """Evaluate pipeline or stage method.

    Parameters
    ----------
    config : Path or str
        Path to config file.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    stages : str, set of str, or None, optional
        Names of stages to evaluate. If None, evaluate entire pipeline.
        Default is None.
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
    model: Pipeline | Stage

    if isinstance(stages, str):
        model = load_stage(config, stages)
        model.evaluate(method, backend, **kwargs)
    else:
        model = load_pipeline(config)
        model.evaluate(method, stages, backend, **kwargs)


def call_function(
    method: Literal["init", "run", "fit", "predict", "collect"], **kwargs
):
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
