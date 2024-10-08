<!--- README template from https://github.com/scottydocs/README-template.md -->
<!-- No pypi or build yet
[![license](https://img.shields.io/pypi/l/OneMod)](https://github.com/ihmeuw-msca/OneMod/blob/main/LICENSE)
[![version](https://img.shields.io/pypi/v/OneMod)](https://pypi.org/project/OneMod)
[![build](https://img.shields.io/github/actions/workflow/status/ihmeuw-msca/OneMod/build.yml?branch=main)](https://github.com/ihmeuw-msca/OneMod/actions)
-->
[![docs](https://img.shields.io/badge/docs-here-green)](https://ihmeuw-msca.github.io/OneMod)
[![codecov](https://img.shields.io/codecov/c/github/ihmeuw-msca/OneMod)](https://codecov.io/gh/ihmeuw-msca/OneMod)
[![codacy](https://img.shields.io/codacy/grade/ae72a07785f5469eac234d1f6bdf555f)](https://app.codacy.com/gh/ihmeuw-msca/OneMod/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


# OneMod

OneMod is a Python package that provides a framework for building and evaluating
single-parameter models. It is designed to be modular, allowing users to easily
add new models and evaluation metrics. OneMod is built on top of the other
IHME tools.

OneMod is being actively developed by the Institute for Health Metrics and is
beta software. It is not yet ready for production use.

For instructions on how to install and use OneMod, please refer to the
[documentation](https://ihmeuw-msca.github.io/OneMod/).

## Pipeline

### Adding stages with dependencies

To add a single stage to a pipeline with dependencies, the stage may be added to the pipeline using the `add_stage` method. This method takes the stage name and the dependencies as arguments. For instance:

```python
# Adding stages programmatically with dependencies
my_pipeline.add_stage(stage_a)
my_pipeline.add_stage(stage_b, dependencies=["stage_a"])
my_pipeline.add_stage(stage_c, dependencies=["stage_a", "stage_b"])
```

To add multiple stages at once with there dependencies, you may use the `add_stages_with_dependencies` method. This method takes a dictionary where the keys are the stage names and the values are the dependencies. For instance:

```python
# Adding stages programmatically with dependencies
stages = {
    "stage_a": [],
    "stage_b": ["stage_a"],
    "stage_c": ["stage_a", "stage_b"]
}
my_pipeline.add_stages_with_dependencies(stages)
```

### Dependency Graph

Dependencies for a given Pipeline instance are defined in the `_dependencies` attribute. This attribute is a dictionary where the keys are the stage names and the values are a set of stage names that the key stage depends on, for instance:

```python
self._dependencies = {
    "stage_b": ["stage_a"],
    "stage_c": ["stage_a", "stage_b"]
}
```

### Pipeline definition with JSON

```json
{
    "name": "example_pipeline",
    "config": {},
    "directory": "/path/to/pipeline_dir",
    "stages": [
        {"name": "stage_a", "config": {}},
        {"name": "stage_b", "config": {}},
        {"name": "stage_c", "config": {}}
    ],
    "dependencies": {
        "stage_b": ["stage_a"],
        "stage_c": ["stage_a", "stage_b"]
    }
}
```

## License

This project uses the following license: [BSD 2-Clause](./LICENSE)
