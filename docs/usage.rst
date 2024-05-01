
Running the pipeline
####################

OneMod is orchestrated with a tool called Jobmon. For a quick intro on Jobmon, refer to the :ref:`jobmon` section.
The full Jobmon documentation lives `here <https://jobmon.readthedocs.io/en/latest/>`_

+++++++++++++++++++++++++++++++
Directory Setup
+++++++++++++++++++++++++++++++

To create a new OneMod pipeline, the first step is setting up your experiment directory. This directory can be in any arbitrary
location, but it must contain a config folder with two files: settings.yml and resources.yml.

The former file, settings.yml, dictates the OneMod parameters you will run your model with. You can specify things like
the parallelism of your model (i.e. by sex/location or by age), the smoothing parameter, the dimensions to smooth over,
covariate constraints, and more.

**TODO: Document sample settings file usage**

The second file, resources.yml, defines the compute resources you will use to run your model. This includes the number of
cores, memory, cluster project, and runtime of your model, and can be broken down by individual task templates. You can
refer to the `Jobmon documentation <https://jobmon.readthedocs.io/en/latest/core_concepts.html#yaml-configuration-files>`_
for more information on how to structure this resources.yml file.

+++++++++++++++++++++++++++++++
Running the pipeline
+++++++++++++++++++++++++++++++

After setting up your experiment directory, running the pipeline is as easy as::

    onemod run_pipeline --directory {path/to/experiment/directory}

The workflow will distribute and monitor jobs until either the workflow is complete, or a job fails.

.. _jobmon-debugging:

+++++++++++++++++++++++++++++++++++
Debugging and resuming the pipeline
+++++++++++++++++++++++++++++++++++

Unfortunately we are not perfect developers, and you are not the perfect user, so sometimes errors occur.

To debug this workflow, you can use the `Jobmon GUI <https://jobmon-gui.scicomp.ihme.washington.edu/>`_.
You can find your workflow by querying for your username, which will bring up the landing page:

.. image:: jobmon_gui_main.png

You can expand a workflow that has a failed status by clicking the workflow ID, bringing up the main workflow page.

.. image:: jobmon_gui_workflow.png

Finally, you can see a traceback by clicking the task template with errors - in this case weave_modeling_template - and
clicking the "Errors" tab. To show a full traceback you can click on an entry in the Error Log field.

.. image:: jobmon_gui_errors.png

Once your errors are fixed, you can resume an existing OneMod pipeline with::

    onemod resume_pipeline --workflow_id {workflow ID}


Both entrypoints (run_pipeline and resume_pipeline) have standard help pages you can access by suffixing the command with -h.


+++++++++++++++++++++++++++++++++++
Output files
+++++++++++++++++++++++++++++++++++

OneMod creates a lot of output files. After running, the experiment directory should look something like this:

.. code-block:: text

    directory/
    |--- config/
    |    |--- settings.yml
    |    |--- resources.yml
    |--- data/
    |    |--- data.parquet
    |--- results/
    |    |--- rover_covsel/
    |    |    |--- coef.pdf
    |    |    |--- selected_covs.yaml
    |    |    |--- subsets.csv
    |    |    |--- summaries.csv
    |    |    |--- submodels/
    |    |    |    |--- subset0/
    |    |    |    |    |--- learner_info.csv
    |    |    |    |    |--- rover.pkl
    |    |    |    |    |--- summary.csv
                    ...
    |    |--- spxmod/
    |    |    |--- coef.csv
    |    |    |--- model.pkl
    |    |    |--- predictions.parquet
    |    |    |--- smooth_coef.pdf
    |    |--- weave/
    |    |    |--- parameters.csv
    |    |    |--- submodels/
    |    |    |    |--- all-age_param0_subset0_holdout1_batch0.parquet
                        ...
    |    |    |--- predictions_holdout0.parquet
    |    |    |--- predictions_holdout1.parquet
    |    |    |    ...

Rover Covsel Files
------------------

The rover_covsel directory contains the output of the covariate selection step.

* selected_covs.yaml

A yaml file containing the covariates selected by the covariate selection step.
Covariates that are significant in at least 50% of submodels are selected in this step for consideration in
the smoothing stage.

* subsets.csv

A csv file indicating what parameters a given subset ID maps to.
Since we can have a large number of groupby parameters, the orchestration layer condenses this to a single
submodel_id parameter. The individual modeling job then reads in subsets.csv to determine what slice of the
data it should be modeling.

* summaries.csv

Covariate coefficients, standard deviation, and overall significance, by subset ID. This is an aggregate of
each submodel summary.csv file.

* coef.pdf

A pdf file containing plots of the selected covariate coefficients. Currently always plotted across age on the x axis.

* submodels/<submodel_id>/learner_info.csv

A serialization of all component learners fit in a single rover submodel. Contains the full set of covariate combinations
explored by that particular rover model, and the scores/weights/coefficient magnitudes for every combination.

* submodels/<submodel_id>/rover.pkl

A fit rover model, containing all explored covariate combinations and their scores.

* submodels/<submodel_id>/summary.csv

Covariate coefficients, standard deviation, and overall significance, by subset ID.

Regmod Smooth Files
-------------------

* coef.csv

Covariate coefficients and standard deviation by smoothing dimension.

* model.pkl

A fit regmod model, containing all explored smoothing dimensions and their scores.

* predictions.parquet

A parquet file containing the predictions of the regmod model.

* smooth_coef.pdf

A pdf file containing plots of the selected covariate coefficients. Currently always plotted across age on the x axis.
Stacks the post-smoothing curves against the pre-smoothing curves from Rover.

Weave Files
-----------

* parameters.csv

A csv file containing the cross product of specified parameters used to fit the weave model.

* predictions_holdout<holdout_id>.parquet

Predictions for a given holdout slice.

* results/weave/submodels/*.parquet

These files contain predictions by subset/parameter/holdout/batch. These are aggregated in the collection stage.
