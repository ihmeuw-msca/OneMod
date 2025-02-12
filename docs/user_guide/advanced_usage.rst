.. _advanced_usage:

Advanced Usage
==============

Input / Output
--------------
The :py:class:`~onemod.io.base.Input` and
:py:class:`~onemod.io.base.Output` classes are dictionary-like objects
that help organize stage dependencies and paths to stage input/output
items. Each stage has an :py:attr:`~onemod.stage.base.Stage.input` and
an :py:attr:`~onemod.stage.base.Stage.output` attribute. Input items,
either paths or upstream stage output, are added to the stage's `input`
attribute when defining pipeline dataflow using the stage's
:py:meth:`~onemod.stage.base.Stage.__call__` method. The `Input` object
checks that all required items have been defined and that all items have
the correct type. Output items, instances of
:py:class:`~onemod.dtypes.data.Data`, are added to the stage's `output`
attribute automatically. These objects contain information about the
output item such as which stage created the item, the path where the
item will be saved, and other metadata (e.g., shape, columns, etc.) used
for validation. The stage's `output` attribute is returned when invoking
the stage's :py:meth:`~onemod.stage.base.Stage.__call__` method when
defining pipeline dataflow:

.. code-block:: python

   preprocessing_output = preprocessing_stage(raw_data="/path/to/raw_data.parquet")
   modeling_output = modeling_stage(observations=preprocessing_output["modeling_data"])
   plotting_output = plotting_stage(
      observations=preprocessing_output["plotting_data"],
      predictions=modeling_output["predictions"],
   )

DataInterface
-------------
The :py:class:`~onemod.fsutils.interface.DataInterface` class is a
unified interface for managing paths to stage input/output and for
loading/dumping files. Each stage has a
:py:attr:`~onemod.stage.base.Stage.dataif` attribute. Paths to the
pipeline's directory and config file and the stage's output directory
are automatically included in the stage's data interface. Paths to stage
input are added to the stage's data interface when defining pipeline
dataflow using the stage's :py:meth:`~onemod.stage.base.Stage.__call__`
method.

* To access paths to stage input/output, use the
  :py:meth:`~onemod.fsutils.interface.DataInterface.get_path` or
  :py:meth:`~onemod.fsutils.interface.DataInterface.get_full_path`
  methods:

  .. code-block:: python

     pipeline_config = stage.dataif.get_path(key="config")
     submodel_dir = stage.dataif.get_full_path("submodels", key="output")

* To load stage input, use the
  :py:meth:`~onemod.fsutils.interface.DataInterface.load` method:

  .. code-block:: python

     observations = stage.dataif.load(key="observations")
     location_metadata = stage.dataif.load("location_metadata.csv", key="data")

* To dump stage output, use the
  :py:meth:`~onemod.fsutils.interface.DataInterface.dump` method:

  .. code-block:: python

     stage.dataif.dump(predictions, "predictions.parquet", key="output")

Creating Custom Stages
----------------------
* required_input, etc.
* skip, collect_after
* implement methods
* create custom config (optional)

..
    Add documentation about validators
