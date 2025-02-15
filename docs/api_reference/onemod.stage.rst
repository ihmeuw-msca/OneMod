onemod.stage
============

.. autoclass:: onemod.stage.base.Stage(name, config=StageConfig(), groupby=None, crossby=None, input_validation=None, output_validation=None)
   :members: get_submodels, from_json, run, fit, predict, collect, __call__

   .. autoproperty:: type() -> str
   .. autoproperty:: module() -> Path | None
   .. autoproperty:: input() -> Input
   .. autoproperty:: output() -> Output
   .. autoproperty:: dependencies() -> list[str]
   .. autoproperty:: dataif() -> DataInterface
   .. autoproperty:: skip() -> list[str]
   .. autoproperty:: subsets() -> DataFrame | None
   .. autoproperty:: paramsets() -> DataFrame | None
   .. autoproperty:: has_submodels() -> bool
   .. autoproperty:: collect_after() -> list[str]
