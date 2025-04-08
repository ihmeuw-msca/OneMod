"""Helper functions for loading custom classes from modules."""

from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmodulename
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel


def get_custom_class(class_name: str, module: str) -> type[BaseModel]:
    """Get custom pipeline, stage, or config class from file.

    Parameters
    ----------
    class_name : str
        Name of custom class.
    module : str
        Path to Python module containing custom class definition.

    Returns
    -------
    BaseModel
        Custom pipeline, stage, or config class.

    """
    loaded_module = load_module(module)
    return getattr(loaded_module, class_name)


def get_custom_config_class(class_name: str, module: str) -> type[BaseModel]:
    """Get custom pipeline config class from file.

    Parameters
    ----------
    class_name : str
        Name of custom pipeline class.
    module : str
        Path to Python module containing custom pipeline class
        definition.

    Returns
    -------
    BaseModel
        Custom pipeline config class from file.

    """
    pipeline_class = get_custom_class(class_name, module)
    config_class = pipeline_class.__pydantic_fields__["config"].annotation
    return config_class


def load_module(module: str) -> ModuleType:
    """Load Python module from file path.

    Parameters
    ----------
    module : str
        Path to Python module.

    Returns
    -------
    ModuleType
        Loaded Python module.

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

    return loaded_module
