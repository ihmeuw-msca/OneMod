onemod.pipeline
===============

.. autoclass:: onemod.pipeline.Pipeline
    :members: from_json, to_json, add_stages, add_stage, build, run, fit, predict

    .. autoproperty:: stages() -> dict[str, Stage]
    .. autoproperty:: dependencies() -> dict[str, list[str]]
