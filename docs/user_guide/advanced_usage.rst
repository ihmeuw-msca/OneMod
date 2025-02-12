.. _advanced_usage:

Advanced Usage
==============

DataInterface
-------------
The :py:class:`~onemod.fsutils.interface.DataInterface` class is a
unified interface for managing paths to stage input/output and for
loading/dumping files. Each stage has a
:py:attr:`~onemod.stage.base.Stage.dataif` attribute. Paths to the
pipeline's directory and config file and the stage's output directory
are automatically included in the stage's data interface. Paths to stage
input are added to the stage's data interface when defining dependencies
using the stage's :py:meth:`~onemod.stage.base.Stage.__call__` method.

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

Input / Output
--------------
* describe what it does, where defined, etc.

Validators
----------
* describe how to use them


Creating Custom Stages
----------------------
* required_input, etc.
* skip, collect_after
* implement methods
* create custom config (optional)
