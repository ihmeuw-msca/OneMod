"""Methods to load and evaluate pipeline and stage objects."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmodulename
from pathlib import Path
from typing import Any, Literal

import fire

import onemod.stage as onemod_stages
from onemod.pipeline import Pipeline
from onemod.stage import Stage


def load_pipeline(config: Path | str) -> Pipeline:
    """Load pipeline instance from JSON file.

    Parameters
    ----------
    config : Path or str
        Path to pipeline config file.

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
        Path to pipeline config file.
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

    Notes
    -----
    When a custom stage class has the same name as a built-in OneMod
    stage class, this function returns the custom stage class.

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
    raise KeyError(
        f"Config does not contain a module for custom stage '{stage_name}'"
    )


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


def evaluate(
    config: Path | str,
    method: Literal["run", "fit", "predict", "collect"] = "run",
    stages: str | list[str] | None = None,
    subsets: dict[str, Any | list[Any]] | None = None,
    paramsets: dict[str, Any | list[Any]] | None = None,
    collect: bool | None = None,
    backend: Literal["local", "jobmon"] = "local",
    cluster: str | None = None,
    resources: Path | str | dict[str, Any] | None = None,
    python: Path | str | None = None,
    **kwargs,
) -> None:
    """Evaluate pipeline or stage method.

    When evaluating a pipeline method, all stage submodels are evaluated
    and submodel results are collected.

    Parameters
    ----------
    config : Path or str
        Path to pipeline config file.
    method : str, optional
        Name of method to evaluate. Default is 'run'.
    stages : str, list of str, optional
        Names of stages to evaluate. If None, evaluate all pipeline
        stages. Default is None.
    backend : str, optional
        Whether to evaluate the method locally or with Jobmon.
        Default is 'local'.
    **kwargs
        Additional keyword arguments passed to stage methods. When
        evaluating a pipeline, use format ``stage={arg_name: arg_value}``.

    Stage Parameters
    ----------------
    subsets : dict, optional
        Submodel data subsets to include when evaluating a single stage.
        If None, evaluate all data subsets. Default is None.
    paramsets : dict, optional
        Submodel parameter sets to include when evaluating a single
        stage. If None, evaluate all parameter sets. Default is None.
    collect : bool, optional
        Whether to collect submodel results when evaluating a single
        stage. If ``subsets`` and ``paramsets`` are both None, default
        is True, otherwise default is False.

    Jobmon Parameters
    -----------------
    cluster : str, optional
        Cluster name. Required if ``backend`` is 'jobmon'.
    resources : Path, str, or dict, optional
        Path to resources file or dictionary of compute resources.
        Required if ``backend`` is 'jobmon'.
    python : Path or str, optional
        Path to Python environment if ``backend`` is 'jobmon'. If None,
        use sys.executable. Default is None.

    Examples
    --------
    This method is the entrypoint when calling onemod from the command
    line:

    .. code:: bash

       onemod --config pipeline_config.json --method run

    To make sure arguments such as lists or dictionaries are parsed
    correctly, omit spaces or use quotes:

    .. code:: bash

       onemod --config pipeline_config.json --stages [stage1,stage2]
       onemod --config pipeline_config.json --stages stage1 --subsets "{'age_group_id': 1}"

    See the `Python Fire Guide <https://google.github.io/python-fire/guide/#argument-parsing>`_
    for more details.

    """
    model: Pipeline | Stage

    if isinstance(stages, str):
        model = load_stage(config, stages)
        model.evaluate(
            method,
            subsets,
            paramsets,
            collect,
            backend,
            cluster,
            resources,
            python,
            **kwargs,
        )
    else:
        model = load_pipeline(config)
        model.evaluate(
            method, stages, backend, cluster, resources, python, **kwargs
        )


def main():
    fire.Fire(evaluate)


if __name__ == "__main__":
    main()
