"""Just something to get my example working."""

import fire
import json
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmodulename
from pathlib import Path
from typing import Literal

from pydantic import validate_call

import onemod.stage as onemod_stages
from onemod.pipeline import Pipeline
from onemod.stage import Stage


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


def load_stage(
    config: Path | str,
    stage_name: str | None = None,
    from_pipeline: bool = False,
) -> Stage:
    """Load stage instance from JSON file.

    Parameters
    ----------
    config : Path or str
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
    stage_class = get_stage(config, stage_name, from_pipeline)
    return stage_class.from_json(config, stage_name, from_pipeline)


def get_stage(
    config: Path | str,
    stage_name: str | None = None,
    from_pipeline: bool = False,
) -> Stage:
    """Get stage class from JSON file.

    Parameters
    ----------
    config : Path or str
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
    with open(config, "r") as f:
        config_dict = json.load(f)
    if from_pipeline:
        try:
            config_dict = config_dict["stages"][stage_name]
        except KeyError:
            raise AttributeError(
                f"{config_dict['name']} does not have a '{stage_name}' stage"
            )
    try:
        if hasattr(onemod_stages, stage_type := config_dict["type"]):
            return getattr(onemod_stages, stage_type)
        else:  # custom stage
            module_path = Path(config_dict["module"])
            spec = spec_from_file_location(
                getmodulename(module_path), module_path
            )
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, stage_type)
    except KeyError:
        raise KeyError(
            "Stage config missing field 'type'; is this a pipeline config file?"
        )


@validate_call
def evaluate(
    config: Path | str,
    stage_name: str | None = None,
    from_pipeline: bool = False,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    backend: Literal["local", "jobmon"] = "local",
    *args,
    **kwargs,
) -> None:
    """Evaluate pipeline or stage method.

    Parameters
    ----------
    config : Path or str
        Path to config file.
    stage_name : str or None, optional
        Stage name, required if `from_pipeline` is True.
        Default is None.
    from_pipeline : bool, optional
        Whether `filepath` is a pipeline or stage config file.
        Default is False.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    backend : str, optional
        Whether to evaluate the method locally or with Jobmon.

    """
    if stage_name is None:
        pipeline = load_pipeline(config)
        pipeline.evaluate(method, backend, *args, **kwargs)
    else:
        stage = load_stage(config, stage_name, from_pipeline)
        stage.evaluate(method, backend, *args, **kwargs)


def main():
    fire.Fire(evaluate)


if __name__ == "__main__":
    main()
