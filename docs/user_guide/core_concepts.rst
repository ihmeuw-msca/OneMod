Core Concepts
=============

Pipeline
--------
The :py:class:`~onemod.pipeline.Pipeline` class contains all of your stages and
their dependencies. After building a pipeline instance, you can evaluate all
stages or a subset of stages by calling pipeline methods
:py:meth:`~onemod.pipeline.Pipeline.run`,
:py:meth:`~onemod.pipeline.Pipeline.fit`, or
:py:meth:`~onemod.pipeline.Pipeline.predict`. Pipeline methods can be called via
the command line (see :py:func:`~onemod.main.evaluate`) or within a script or
notebook, and they can be evaluated either locally or using
`Jobmon <https://jobmon.readthedocs.io/en/latest/index.html>`_. Pipeline
instances are saved as a JSON file.

Dataflow
^^^^^^^^
After adding stages to your pipeline, you need to define the "dataflow", i.e.,
how data is passed from one stage to another, using the stage
:py:meth:`~onemod.stage.base.Stage.__call__` method. This section of code
defines the stage dependencies that are used to create the pipeline's directed
acyclic graph (DAG) of tasks:

.. code-block:: python

   preprocessing_output = preprocessing_stage(raw_data="/path/to/raw_data.parquet")
   modeling_output = modeling_stage(observations=preprocessing_output["modeling_data"])
   plotting_output = plotting_stage(
       observations=preprocessing_output["plotting_data"],
       predictions=modeling_output["predictions"],
   )

When defining the dataflow, information about input/output data is passed from
stage to stage (e.g., paths to where output will be saved), but the output is
not created until the stages' :py:meth:`~onemod.stage.base.Stage.run`,
:py:meth:`~onemod.stage.base.Stage.fit`, or
:py:meth:`~onemod.stage.base.Stage.predict` methods are evaluated.

Stage
-----
The :py:class:`~onemod.stage.base.Stage` class contains attributes and methods
corresponding to a pipeline stage. Users can implement their own custom stages
(see :ref:`Advanced Usage`) or create instances of the model stages available
within the **OneMod** package.

skip
^^^^
While all stages are required to implement the
:py:meth:`~onemod.stage.base.Stage.run` method, they may skip the
:py:meth:`~onemod.stage.base.Stage.fit` or
:py:meth:`~onemod.stage.base.Stage.predict` methods. The
:py:attr:`~onemod.stage.base.Stage.skip` attribute contains the names of any
methods skipped by a stage class.

groupby / subsets
^^^^^^^^^^^^^^^^^
The :py:attr:`~onemod.stage.base.Stage.groupby` attribute allows stages to be
parallelized over subsets of your input data. For example, a stage can evaluate
a separate submodel for each age group by setting ``groupby = ['age_group_id']``
when defining the stage instance. Subsets will be created based on the
``age_group_id`` column in the pipeline's
:py:attr:`~onemod.pipeline.Pipeline.groupby_data` attribute, and they can be
accessed via the stage's :py:attr:`~onemod.stage.base.Stage.subsets` attribute.

crossby / paramsets
^^^^^^^^^^^^^^^^^^^
The :py:attr:`~onemod.stage.base.Stage.crossby` attribute allows stages to be
parallelized over different parameter values. For example, a stage can evaluate
separate submodels for different parameter values or holdout sets by setting
``crossby = ['param', 'holdout']`` when defining the stage instance. Parameter
sets will be created based on the ``param`` and ``holdout`` values defined in
the stage's :py:attr:`~onemod.stage.base.Stage.config` attribute, and they can
be accessed via the stage's :py:attr:`~onemod.stage.base.Stage.paramsets`
attribute.

submodels
^^^^^^^^^
Each stage submodel corresponds to a single ``subset`` / ``paramset``
combination. For a list of all submodels corresponding to a stage instance, use
the :py:meth:`~onemod.stage.base.Stage.get_submodels` method. The stage methods
:py:meth:`~onemod.stage.base.Stage.run`,
:py:meth:`~onemod.stage.base.Stage.fit`, and
:py:meth:`~onemod.stage.base.Stage.predict`, can be evaluated for a single
submodel, a subset of submodels, or all submodels. For example, if a stage's
submodels vary by age and location, you can run different combinations of
submodels using the ``subsets`` argument:

.. code-block:: python

   stage.run(subsets={"age_group_id": 1, "location_id": 1})  # single submodel
   stage.run(subsets={"age_group_id": [1, 2]})  # two age groups, all locations
   stage.run()  # all submodels

When evaluating the pipeline's :py:meth:`~onemod.pipeline.Pipeline.run`,
:py:meth:`~onemod.pipeline.Pipeline.fit`, or
:py:meth:`~onemod.pipeline.Pipeline.predict` methods, all submodels are always
evaluated and submodel output collected.

collect_after
^^^^^^^^^^^^^
Stages with submodels have the option to collect submodel output after
the :py:meth:`~onemod.stage.base.Stage.run`,
:py:meth:`~onemod.stage.base.Stage.fit`, or
:py:meth:`~onemod.stage.base.Stage.predict` methods are evaluated. For example,
stages using the :py:attr:`~onemod.stage.base.Stage.groupby` attribute might
concatenate the predictions corresponding to each data subset, or stages using
the :py:attr:`~onemod.stage.base.Stage.crossby` attribute might ensemble the
predictions corresponding to each parameter set based on out-of-sample
performance. The stage :py:attr:`~onemod.stage.base.Stage.collect_after`
attribute contains the names of any methods that require submodel collection via
the stage's :py:meth:`~onemod.stage.base.Stage.collect` method. When calling a
method in :py:attr:`~onemod.stage.base.Stage.collect_after`, you can control
whether or not submodel output is collected using the ``collect`` argument. When
evaluating all submodels, the default is to collect submodel output, otherwise
the default is not to collect submodel output.

Config / StageConfig
--------------------
The :py:class:`~onemod.config.base.Config` and
:py:class:`~onemod.config.base.StageConfig` classes are dictionary-like objects
that contain pipeline and/or stage settings. For settings validation via
`Pydantic <https://docs.pydantic.dev/latest/>`_, users can create custom
configuration classes (see :ref:`Advanced Usage`). Stage
:py:attr:`~onemod.stage.base.Stage.config` attributes automatically have access
to the settings within their corresponding pipeline's
:py:attr:`~onemod.pipeline.Pipeline.config` attribute, which allows you to treat
pipeline config items like global settings.
