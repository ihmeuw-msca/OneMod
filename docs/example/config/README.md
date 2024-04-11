## Settings

The [settings](settings.yml) file contains the OneMod stage model settings.

### OneMod settings

Parameters:
- `input_path` (str) - Path to input data. Must be a parquet file.
- `col_id` (str or list of str) - Column name(s) for point IDs. For example, age_group_id, location_id, sex_id, and/or year_id, etc. To run a OneMod pipeline on a subset of the input data, include a list or lists of the ID values to use in the model.
- `col_obs` (str) - Column name for input observations.
- `col_pred` (str) - Column name for OneMod predictions.
- `col_holdout` (str or list or str) - Column name(s) for OneMod stage out-of-sample performance. Values can be either 0 (training data), 1 (holdout data), or NaN (missing input observations).
- `col_test` (str) - Column name for OneMod pipeline out-of-sample performance. Values can be either 0 (training data), 1 (testing data), or NaN (missing input observations).

### OneMod stage settings

Parameters:
- `max_attempts` (int) - Number of attempts to try stage tasks before giving up.
- `groupby` (str or list of str, optional) - Column name(s) for model subsets. For example, separate models can be run for each age_group_id, sex_id and/or super_region_id, etc.

**Rover settings**

See the [ModRover documentation](https://ihmeuw-msca.github.io/modrover/) for a full description of the Rover stage parameters. 

Optional parameters:
- `col_offset` (str, optional) - Column name for model offset values. If not specified, default offset value is 0.
- `col_weights` (str, optional) - Column name for model weight values. If not specified, default weight value is 1.

**WeAve settings**

See the [WeAve documentation](https://ihmeuw-msca.github.io/weighted-average/) for a full description of the WeAve stage parameters. Dimensions using the `depth` kernel (e.g., `location_id`) should be listed last for weight normalization to work as intended.

Optional parameters:
- `max_batch` (int, optional) - Maximum number of prediction points per WeAve model. Reduces memory requirements for models with large weight matrices.

For the WeAve stage, you can include multiple model configurations (note: model names cannot contain underscores). In addition, you can specify a list of values for the following model parameters:
- `radius`
- `exponent`
- `distance_dict`

**Ensemble settings**

Parameters:
- `metric` {rmse} - Performance metric for smoother out-of-sample results.
- `score` {avg, rover, codem, best} - Score function to convert performance metrics to weight values; avg returns the average of all smoother submodel predictions, rover uses the ensemble method from the ModRover package, codem uses the ensemble method from CODEm, and best returns the predictions from the smoother submodel with the best average out-of-sample performance.
- `top_pct_score` (float in (0,1], optional) - If `score` is rover, ensemble only the models with scores within `top_pct_score`% of the highest submodel score.
- `top_pct_model` (float in (0, 1], optional) - If `score` is rover, ensemble only the top `top_pct_model`% models by score.
- `psi` (float, optional) - Score function parameter if `score` is codem.

## Resources

The optional [resources](resources.yml) file contains OneMod stage cluster resources. If not included, Jobmon default resources are used.

Parameters:
- `tool_resources` (required) - Default resources for all pipeline tasks.
- `initialization_template`, `rover_submodel_initialization_template`, (optional) - Resources for stage initialization tasks (i.e., deleting previous results and initializing result directories).
- `rover_modeling_template`, `weave_modeling_template`, `ensemble_modeling_template` (optional) - Resources for stage modeling tasks.
- `collection_template` (optional) - Resources for stage collection tasks (i.e., collecting submodel results).
- `deletion_template`, `rover_submodel_deletion_template`, (optional) - Resources for stage deletion tasks (i.e., deleting intermediate results if `save_intermediate` is False).
