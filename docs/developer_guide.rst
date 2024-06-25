Developer's Guide
=================

Code Layout
-----------

For the most part, all code run by this repository lies under ``src/onemod``. Within that repository:

- ``main.py`` is the core entrypoint, that exposes the functions users interact with directly. 
- ``stages.py`` defines a concept of a Stage, that can dynamically create Jobmon task templates and spawn tasks from them
- ``model/`` contains a set of Python scripts that perform the data transformations. 
- ``data/`` contains a set of utilities mainly related to file system management (directory creation/deletion, concatenating results, etc.)

Orchestration
-------------

At its core, this repository is a data pipeline that sequentially performs a series of transformations on an input dataset.
For large datasets, sequential or parallel computations necessitate some kind of automated orchestration to chunk the 
computation and parallelize over a distributed cluster. In onemod, this orchestration layer is Jobmon. 


.. _jobmon:

Jobmon
~~~~~~

As a 30 second introduction to Jobmon, it's an orchestration module, written in Python, that allows you to define a 
**workflow**, create **tasks** to add to that workflow, set dependencies on its tasks, and run the workflow.

A **workflow** is a computational graph of work to be done, the building blocks of the graph are **tasks**. Tasks contain
a bash command which is the command to run when that task is scheduled to execute. After building tasks, setting dependencies,
and adding to a workflow, all the user needs to do is call the ``Workflow.run()`` method to begin execution of the workflow.
At this point the user can sit back and monitor the progress without needing to wait around for intermediate phases to conclude - 
Jobmon will wait until a task completes and automatically schedule the downstream tasks in the graph.

There is an intermediate concept called ``TaskTemplate`` - the most direct way of thinking about task templates is that they
are simply
a command template that you can parametrize with keyword args to create tasks. For example, if you have a task that prints out
an arbitrary message, you can create a task template with the command template ``echo {message}``. A task will be an instance
of this template, i.e. ``Task(message='hello'), Task(message='world')``.

In OneMod terms, a workflow is equivalent to a ``pipeline`` - i.e. the ``onemod run_pipeline`` entrypoint creates a new workflow.
A task is equivalent to a unit of work, ie. ``rover_covsel_model --submodel_id 0``.

You can set dependencies between tasks, so that a task that needs some output file from a prior task will wait until that prior task completes.
With onemod as an example, the ``collect_results rover_covsel`` task will wait until all ``rover_covsel_model`` tasks have
completed - without dependencies, they would fail immediately since the expected output files would not exist.


You can monitor the progress of a Jobmon workflow using the `Jobmon GUI <https://jobmon-gui.scicomp.ihme.washington.edu/>`_, 
a task that has an error will block subsequent tasks from executing and report the error message so you can debug and fix. 
An additional feature of Jobmon is that if you encounter and error and fix the bug, it's simple to resume the workflow
from the last point of failure, saving the tedium of re-running previous steps that have already completed unnecessarily.

For more details on Jobmon, please refer to the `full documentation <https://jobmon.readthedocs.io/en/latest/>`_. If you're curious
about the detailed debugging process useful for OneMod, refer to the :ref:`Jobmon debugging <jobmon-debugging>` section.

Subsets and Submodels
~~~~~~~~~~~~~~~~~~~~~

A key requirement of OneMod is the ability to flexibly model different sets of fixed/random effects. To facilitate computation,
we need to be able to split up the data across different axes arbitrarily. The concept of subsets exists to work nicely with different
chunks of data containing different identifying attributes. 

The ``groupby`` parameters set in the settings.yml file identify subgroups in the input dataset - e.g. a value of 
``[year_id, sex_id]`` indicates that rows are uniquely identified by year and sex and can thus be modeled independently. 

However, data volume is not always evenly distributed across the groups - certain years or locations can contain more data
than other groupings. To enforce smaller groups and thus quicker computation, we can further split up a ``submodel`` into 
additional ``subsets`` (without doing this, we'll be waiting unnecessarily for large subsets to complete fitting and might
run into memory issues with those larger subsets). 

Unit Testing
------------

The ``tests/`` folder of this repository contain a series of unit tests, at this moment mainly concerned with unit testing
the orchestration layer. This project uses the ``pytest`` framework for running unit tests. 

You can run the test suite using ``nox``, a Python package that manages virtual environments for testing purposes. The command
is simply ``nox -r -s tests``. Once run, nox will create a conda environment at ``.nox/tests`` relative to the root of the repository.
As a time saving measure, if you're running repeated tests, you can activate this environment and call ``pytest`` yourself.

Common Errors and Solutions
---------------------------

- Configuration errors

  OneMod uses a tool called ``pydantic`` to validate the settings.yml file. If you encounter an error
  that looks like ``ValidationError: 1 validation error for Settings`` it's likely that you've made some kind of error in
  creating your settings.yml file. The error message should indicate the line number and the specific error.

  - Most of the time, the error is a missing parameter or an incorrect indent. For example, ``groupby`` will be a required parameter at the top
    level of the settings.yml file, so if you forget to include it, you'll get an error.

  - Another common error cause is specifying important columns as ID columns, holdouts, observations, or covariates when they
    aren't present in the dataset.

      - **TODO** OneMod should actually check for these errors and raise a pydantic exception. As of right now, this type of error
        will simply manifest as a KeyError in some downstream process, potentially after the workflow starts running.

- Exploding Covariates and Incorrect Model Selection

  A model_type like binomial nominally expects rates data. If your data is not properly converted to a rate or is not
  normalized, Rover or Regmod Smooth could fail. You'll first see warnings like::

    /ihme/homes/dhs2018/miniconda3/envs/onemod/lib/python3.11/site-packages/regmod/models/binomial.py:81: RuntimeWarning: divide by zero encountered in divide
      (param - self.data.obs) / (param*(1 - param)) * dparam
    /ihme/homes/dhs2018/miniconda3/envs/onemod/lib/python3.11/site-packages/regmod/models/binomial.py:81: RuntimeWarning: invalid value encountered in divide
      (param - self.data.obs) / (param*(1 - param)) * dparam
    /ihme/homes/dhs2018/miniconda3/envs/onemod/lib/python3.11/site-packages/regmod/models/binomial.py:81: RuntimeWarning: invalid value encountered in multiply
      (param - self.data.obs) / (param*(1 - param)) * dparam

  The root cause is that the ``param`` variable is calculated with an inverse link function, usually something like ``expit``.
  For very large values this function goes to 0, leading to divide by 0 errors.

  The simplest way to fix is to use a counts model like ``gaussian`` or ``poisson``, or normalize your data to a rate.
  OneMod also allows you to add coefficient boundaries in the settings file.

- No data in a given subset

  For parallelization and cross validation, data is usually split up into a product of your groupby parameters and your specified
  holdout columns. The holdout columns are assumed to have been generated randomly or non-randomly  by the user in some upstream process,
  and if generated incorrectly (or unluckily) it's possible some subsets will have no training data or no test data.

  This can cause errors in Rover or Weave. The fix must be upstream; whatever mechanism used to generate holdouts, every holdout
  column must have at least 1 ``1`` and at least 1 ``0`` per groupby parameter.

Architecture
------------

OneMod uses the Model-View-Controller (MVC) design pattern.
The architecture of OneMod is designed to be modular and extensible. The core of the project is the ``model/`` directory, which
contains a series of Python scripts that define the transformations to be applied to the input data. These transformations
are designed to be as general as possible, so that they can be applied to a wide variety of datasets.


Documentation and Deployment
----------------------------

TODO