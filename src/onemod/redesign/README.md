# OneMod Redesign

*GENERAL NOTES PENDING REDESIGN FILES RESTRUCTURING*

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
