# OneMod Project Directory Structure

Ideas for how a OneMod project directory structure could be organized. Would be initialized with a command like `onemod init` or `onemod create` or `onemod new` or `onemod startproject` or similar.

## Structure Ideas

```text
project_name/
    stages/
        custom_data_prep.py
        custom_spxmod.py
    config/
        config.yaml
    experiments/
        config/
            stage1/
                config.yaml
            pipeline_config.yaml
            run_config.yaml
    scripts/
        run.py
    logs/
        errors.log
    README.md
```
