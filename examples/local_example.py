"""Run OneMod pipeline locally using OneMod."""

from itertools import product

import fire

from onemod import Pipeline


def run_pipeline(config_json: str, method: str = "run") -> None:
    # Load pipeline
    example_pipeline = Pipeline.from_json(filepath=config_json)

    # Run tasks
    for stage in example_pipeline.stages.values():
        if method not in stage._skip_if:
            subset_ids = getattr(stage, "subset_ids", None)
            param_ids = getattr(stage, "param_ids", None)
            if subset_ids is not None or param_ids is not None:
                for subset_id, param_id in product(
                    subset_ids or [None], param_ids or [None]
                ):
                    stage.evaluate(
                        method=method, subset_id=subset_id, param_id=param_id
                    )
                stage.collect()
            else:
                stage.evaluate(method=method)


if __name__ == "__main__":
    fire.Fire(run_pipeline)
