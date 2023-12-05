
Running the pipeline
####################

OneMod is orchestrated with a tool called Jobmon. You can reference the full Jobmon documentation `here <https://jobmon.readthedocs.io/en/latest/>`_.

The short version is that Jobmon allows us to define a modeling pipeline for distributing jobs on IHME's SLURM cluster.
Jobs are run in parallel where possible, with automatic dependency management, so all you have to do is call the entrypoint
and wait for results. An additional benefit of Jobmon is that if you have a data error or config error, you can fix it and
resume from the last point of failure.

To create a new OneMod pipeline::

    onemod run_pipeline --directory {path/to/experiment/directory}

The workflow will distribute and monitor jobs until either the workflow is complete, or a job fails. In the latter case,
to debug this workflow, you should use the `Jobmon GUI <https://jobmon-gui.scicomp.ihme.washington.edu/>`_. You can find your workflow
by querying for your username, and then clicking on the workflow ID -> Task Template bar with failures -> Errors Tab to
surface your errors.

Once your errors are fixed, you can resume an existing OneMod pipeline with::

    onemod resume_pipeline --workflow_id {workflow ID}


Both entrypoints (run_pipeline and resume_pipeline) have standard help pages you can access by suffixing the command with -h.
