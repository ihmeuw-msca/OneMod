Advanced Usage
==============

Data
----
The :py:class:`~onemod.dtypes.Data` class contains information about input /
output items such as which stage and method they correspond to, the path where
the item will be saved, and other metadata (e.g., format, shape, columns, etc.)
used for validation.

Input / Output
--------------
The :py:class:`~onemod.io.base.Input` and :py:class:`~onemod.io.base.Output`
classes are dictionary-like objects that help organize and validate stage
dependencies, instances of :py:class:`~onemod.dtypes.data.Data`. Each stage has
an :py:attr:`~onemod.stage.base.Stage.input` and an
:py:attr:`~onemod.stage.base.Stage.output` attribute. Input items, either paths
or upstream stage output, are added to the stage's ``input`` attribute when
defining pipeline dataflow. Output items are added to the stage's ``output``
attribute automatically. The stage's ``output`` attribute is returned when
invoking the stage's :py:meth:`~onemod.stage.base.Stage.__call__` method when
defining pipeline dataflow.

DataInterface
-------------
The :py:class:`~onemod.fsutils.interface.DataInterface` class is a unified
interface for managing paths to stage input/output and for loading/dumping
files. Each stage has a :py:attr:`~onemod.stage.base.Stage.dataif` attribute.
Paths to the pipeline's directory and config file and the stage's output
directory are automatically included in the stage's data interface. Paths to
stage input are added to the stage's data interface when defining pipeline
dataflow.

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

Validation
----------
The optional stage attributes
:py:attr:`~onemod.stage.base.Stage.input_validation` and
:py:attr:`~onemod.stage.base.Stage.output_validation` allow users to specify
features of their input/output data that require validation. See
:py:class:`~onemod.dtypes.Data` for more details.

Customization
-------------
The :py:class:`~onemod.pipeline.Pipeline`,
:py:class:`~onemod.config.base.Config`, and
:py:class:`~onemod.config.base.StageConfig` classes can be used as-is, but you
may want to write your own subclasses to add custom attributes or methods. On
the other hand, the :py:class:`~onemod.stage.base.Stage` class is an abstract
class, so you will need to implement your own subclass or use the model stages
available within the **OneMod** package.

Stage Classes
^^^^^^^^^^^^^
When creating a custom stage, you should include the following private
attributes: ``_required_input``, ``_optional_input``, ``_output_items``. These
attributes are dictionaries with keys corresponding to input/output items and
values containing information about the item in the form of a dictionary. Values
must include the expected file format (or "directory"), but do not include a
path. Values can also contain metadata (e.g., shape, columns, etc.) used for
valiation. If you have method-specific dependencies, you can also specify which
method uses or creates the item. See :py:class:`~onemod.dtypes.Data` for more
details.

* Input items that come from an upstream stage must be included in either
  ``_required_input`` or ``_optional_input`` and defined in the dataflow to be
  added to the stage's :py:attr:`~onemod.stage.base.Stage.dependencies`
  attribute. Items defined in the dataflow are also added to the stage's
  :py:attr:`~onemod.stage.base.Stage.dataif` attribute for easy loading. All
  items in ``_required_input`` must be defined in the pipeline's dataflow. Items
  defined in the dataflow but not in either ``_required_input`` or
  ``_optional_input`` are ignored.
* Input items that do not come from an upstream stage do not need to be included
  in ``_required_input`` or ``_optional_input`` (e.g., they could be specified
  in the stage's :py:attr:`~onemod.stage.base.Stage.config` attribute), but they
  must be included if they require validation.
* Stages can create output items that are not included in ``_output_items``,
  but items must be included if they will be passed to downstream stages or if
  they require any validation.

Stage classes are required to implement the ``_run()`` method. They do not need
to implement the ``_fit()`` or ``_predict()`` methods. These private methods are
meant to evaluate a single submodel. When omitting any of these methods, be sure
to include their name(s) in the stage's private ``_skip`` attribute. This
ensures that the stage is skipped when calling the pipeline's
:py:meth:`~onemod.pipeline.Pipeline.fit` or
:py:meth:`~onemod.pipeline.Pipeline.predict` method. Stage methods
:py:meth:`~onemod.stage.base.Stage.run`,
:py:meth:`~onemod.stage.base.Stage.fit`, and
:py:meth:`~onemod.stage.base.Stage.predict`, which are meant to run all
submodels or a subset of submodels, should not be modified.

For a stage to use the :py:attr:`~onemod.stage.base.Stage.groupby` attribute,
you must include the ``subset`` argument to the ``_run()``, ``_fit()``, and
``_predict()`` methods. Similarly, for a stage to use the
:py:attr:`~onemod.stage.base.Stage.crossby` attribute, you must include the
``paramset`` argument. You can also include any custom keyword arguments you
like.

To collect submodel output when evaluating the stage's
:py:meth:`~onemod.stage.base.Stage.run`,
:py:meth:`~onemod.stage.base.Stage.fit`, or
:py:meth:`~onemod.stage.base.Stage.predict` methods, include the method name(s)
in the stage's private ``_collect_after`` attribute. This ensures that the
stage's :py:meth:`~onemod.stage.base.Stage.collect` method is called after all
submodels have been evaluated. You must also implement the
:py:meth:`~onemod.stage.base.Stage.collect` method.

Configuration Classes
^^^^^^^^^^^^^^^^^^^^^
You can pass any setting to existing :py:class:`~onemod.config.base.Config` or
:py:class:`~onemod.config.base.StageConfig` classes without creating your own
subclasses. However, creating your own subclasses allows you to add validation.
Both configuration classes are subclasses of Pydantic's
`BaseModel <https://docs.pydantic.dev/latest/api/base_model/>`_ class. By adding
your own model fields with type hints, your custom configuration class will
automatically validate any user-supplied settings.

* Stage :py:attr:`~onemod.stage.base.Stage.config` attributes have access to
  their corresponding pipeline's :py:attr:`~onemod.pipeline.Pipeline.config`
  attribute. For example, ``stage.config["id_columns"]`` will return
  ``id_columns`` from the stage's config if it exists and is not ``None``,
  otherwise it will return ``id_columns`` from the pipeline's config if it
  exists and is not ``None``.
* If a stage has a required setting that can be specified at either the stage or
  pipeline level, the item should include ``None`` as its default in the custom
  stage config and the item's name should be included in the stage config's
  ``_required`` attribute.
* To enable the :py:attr:`~onemod.stage.base.Stage.crossby` attribute for a
  setting in a custom stage config, the setting's type hints must include a
  list, set, or tuple. For example, ``param: int | list[int]``.
