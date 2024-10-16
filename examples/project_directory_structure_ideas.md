# OneMod Project Directory Structure

Ideas for how a OneMod project directory structure could be organized. Would be initialized with a command like `onemod init` or `onemod create` or `onemod new` or `onemod startproject` or similar.

## Structure Ideas

```text
project_name/
    stages/
        custom_data_prep.py
        custom_spxmod.py
        kreg/
            kernel_functions/
            custom_kreg.py
    config/
        pipeline_config.json
        jobmon_config.json
        airflow_config.json
    pipeline_definition.py
    logs/
        errors.log
    README.md
```
