.. role:: python(code)
   :language: python

Core Concepts
=============

Pipeline
--------
The :py:class:`~onemod.pipeline.Pipeline` class contains all of your
stages and their dependencies. After building a pipeline instance, you
can evaluate all stages or a subset of stages by calling pipeline
methods :py:meth:`~onemod.pipeline.Pipeline.run`,
:py:meth:`~onemod.pipeline.Pipeline.fit`, or
:py:meth:`~onemod.pipeline.Pipeline.predict`. Pipeline methods can be
called via the command line (see :py:func:`~onemod.main.evaluate`) or
within a script or notebook, and they can be evaluated either locally or
using `Jobmon <https://jobmon.readthedocs.io/en/latest/index.html>`_.
Pipeline instances are saved as a JSON file.

Stage
-----
The :py:class:`~onemod.stage.base.Stage` class contains attributes and
methods corresponding to a pipeline stage. Users can implement their
own custom stages (see :ref:`advanced_usage`) or create instances of the
model stages available within the OneMod package.

skip
^^^^
While all stages are required to implement the
:py:meth:`~onemod.stage.base.Stage.run` method, they may skip the
:py:meth:`~onemod.stage.base.Stage.fit` or
:py:meth:`~onemod.stage.base.Stage.predict` methods. The
:py:attr:`~onemod.stage.base.Stage.skip` attribute contains the names of
any methods skipped by a stage class.

groupby / subsets
^^^^^^^^^^^^^^^^^
The :py:attr:`~onemod.stage.base.Stage.groupby` attribute allows stages
to be parallelized over subsets of your input data. For example, a stage
can evaluate a separate submodel for each age group by setting
:python:`groupby = ['age_group_id']` when defining the stage instance.
Subsets will be created based on the `age_group_id` column of your input
data, and they can be accessed via the stage's
:py:attr:`~onemod.stage.base.Stage.subsets` attribute.

crossby / paramsets
^^^^^^^^^^^^^^^^^^^
The :py:attr:`~onemod.stage.base.Stage.crossby` attribute allows stages
to be parallelized over different parameter values. For example, a stage
can evaluate separate submodels for different parameter values or
holdout sets by setting :python:`crossby = ['param', 'holdout']` when
defining the stage instance. Parameter sets will be created based on the
`param` and `holdout` values defined in the stage's
:py:attr:`~onemod.stage.base.Stage.config` attribute, and they can be
accessed via the stage's :py:attr:`~onemod.stage.base.Stage.paramsets`
attribute.

submodels
^^^^^^^^^
Each stage submodel corresponds to a single `subset` / `paramset`
combination. For a list of all submodels corresponding to a stage
instance, use the :py:meth:`~onemod.stage.base.Stage.get_submodels``
method.

collect_after
^^^^^^^^^^^^^
Stages with submodels have the option to collect submodel output after
the :py:meth:`~onemod.stage.base.Stage.run`,
:py:meth:`~onemod.stage.base.Stage.fit`, or
:py:meth:`~onemod.stage.base.Stage.predict` methods are evaluated. For
example, stages using the :py:attr:`~onemod.stage.base.Stage.groupby`
attribute might concatenate the predictions corresponding to each data
subset, or stages using the :py:attr:`~onemod.stage.base.Stage.crossby`
attribute might ensemble the predictions corresponding to each parameter
set based on out-of-sample performance. The stage
:py:attr:`~onemod.stage.base.Stage.collect_after` attribute contains the
names of any methods that require submodel collection via the stage's
:py:meth:`~onemod.stage.base.Stage.collect` method.

Config / StageConfig
--------------------
The :py:class:`~onemod.config.base.Config` and
:py:class:`~onemod.config.base.StageConfig` classes are dictionary-like
objects that contain pipeline and/or stage settings. For settings
validation via `Pydantic <https://docs.pydantic.dev/latest/>`_, users
can create custom configuration classes. Stage
:py:attr:`~onemod.stage.base.Stage.config` attributes have access to
the settings within their corresponding pipeline's
:py:attr:`~onemod.pipeline.Pipeline.config` attribute.
