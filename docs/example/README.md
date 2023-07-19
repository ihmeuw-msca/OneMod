## Instructions

**Activate your OneMod environment:**
```
conda activate {environment name}
```

**Create a new OneMod pipeline:**
```
onemod run_pipeline --directory {path/to/experiment/directory}
```

Parameters:
- `directory` (str) - Path to experiment directory. Must contain config/settings.yml.
- `stages` (str or list of str, optional) - Pipeline stage(s) to run. Default is [rover,swimr,weave,ensemble].
- `save_intermediate` (bool, optional) - Save intermediate stage results. Default is False.
- `cluster_name` (str, optional) - Name of cluster to run pipeline on. Default is slurm.
- `configure_resources` (bool, optional) - Configure resources in directory/config/resources.yml. Default is True.

**Resume an existing OneMod pipeline:**
```
onemod resume_pipeline --workflow_id {workflow ID}
```

Parameters:
- `workflow_id` (int) - ID of the workflow to resume.
- `cluster_name` (str, optional) - Name of cluster to run pipeline on. Default is slurm.

**Additional entrypoints:**
- `initialize_results {stage} --experiment_dir {path/to/experiment/directory}`
- `collect_results {stage} --experiment_dir {path/to/experiment/directory}`
- `delete_results {stage} --experiment_dir {path/to/experiment/directory}`
- `rover_model --experiment_dir {path/to/experiment/directory} --submodel_id {submodel ID}`
- `swimr_model --experiment_dir {path/to/experiment/directory} --submodel_id {submodel ID}`
- `weave_model --experiment_dir {path/to/experiment/directory} --submodel_id {submodel ID}`
- `ensemble_model --experiment_dir {path/to/experiment/directory}`

### Input

The experiment directory must have a config directory with the following
structure:

```
experiment/
- config/
   - settings.yml
   - resources.yml (optional)
```

See [config](config/) for example configuration files.

### Output

Stage output is saved in the results directory, and any previous results for the selected stages will be deleted.
For each stage, tables are created to match subset IDs with subset column values.
For each WeAve and SWiMR model configuration, a table is saved to match parameter IDs with model parameters.
If `save_intermediate` is True, intermediate results are saved in each model directory for debugging.

```
experiment/
- results/
  - rover/
    - predictions.parquet
    - subsets.csv
  - swimr/
    - predictions_holdout1.parquet
    - ...
    - predictions_holdout5.parquet
    - predictions.parquet
    - parameters.csv
    - subsets.csv
  - weave/
    - predictions_holdout1.parquet
    - ...
    - predictions_holdout5.parquet
    - predictions.parquet
    - parameters.csv
    - subsets.csv
  - ensemble/
    - predictions.parquet
    - performance.csv
    - subsets.csv
```
