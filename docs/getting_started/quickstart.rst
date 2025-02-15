Quickstart
==========

Define your pipeline:

.. code-block:: python

   pipeline = Pipeline(
       name="example_pipeline",
       directory="/path/to/pipeline_directory/",
       config={"ids": ["age_group_id", "sex_id", "year_id", "location_id"]},
       groupby_data="/path/to/groupby_data.parquet",
    )

Define your stages:

.. code-block:: python

   preprocessing_stage = PreprocessingStage(name="preprocessing_stage")
   modeling_stage = ModelingStage(
       name="modeling_stage",
       config={
           "param": [1, 2],
           "holdout": ["holdout1", "holdout2"],
       },
       groupby=["age_group_id"],
       crossby=["param", "holdout"]
   )
   plotting_stage = PlottingStage(
       name="plotting_stage",
       config={"x": ["age_group_id", "year_id"]},
       groupby=["location_id"],
       crossby=["x"],
   )

Define dataflow:

.. code-block:: python

   preprocessing_output = preprocessing_stage(raw_data="/path/to/raw_data.parquet")
   modeling_output = modeling_stage(observations=preprocessing_output["modeling_data"])
   plotting_output = plotting_stage(
       observations=preprocessing_output["plotting_data"],
       predictions=modeling_output["predictions"],
   )

Build pipeline and save to JSON file:

.. code-block:: python

   pipeline.build()

Run pipeline:

.. code-block:: python

   pipeline.run(
       backend="jobmon",
       cluster="cluster_name",
       resources="/path/to/compute_resources.json",
   )
