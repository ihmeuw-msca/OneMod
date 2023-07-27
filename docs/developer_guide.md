# Developer's Guide

## Code Layout

For the most part, all code run by this repository lies under `src/onemod`. Within that repository:
- `main.py` is the core entrypoint, that exposes the functions users interact with directly. 
- `stages.py` defines a concept of a Stage, that can dynamically create Jobmon task templates and spawn tasks from them
- `model/` contains a set of Python scripts that perform the data transformations. 
- `data/` contains a set of utilities mainly related to file system management (directory creation/deletion, concatenating results, etc.)

## Orchestration

At its core, this repository is a data pipeline that sequentially performs a series of transformations on an input dataset.
For large datasets, sequential or parallel computations necessitate some kind of automated orchestration to chunk the 
computation and parallelize over a distributed cluster. In onemod, this orchestration layer is Jobmon. 

### Jobmon

As a 30 second introduction to Jobmon, it's an orchestration module, written in Python, that allows you to define a 
**workflow**, create **tasks** to add to that workflow, set dependencies on said tasks, and run the workflow. 

A workflow is a computational graph of work to be done, the building blocks of said graph are jobmon Tasks. Tasks contain
a bash command indicating the command to run when said task is scheduled to execute. After building tasks, setting dependencies, 
and adding to a workflow, all the user needs to do is call the `Workflow.run()` method to begin execution of said workflow. 
At this point the user can sit back and monitor the progress without needing to wait around for intermediate phases to conclude - 
Jobmon will wait until a task completes and automatically schedule the downstream tasks in the graph. 

You can monitor the progress of a Jobmon workflow using the [Jobmon GUI](https://jobmon-gui.scicomp.ihme.washington.edu/), 
a task that has an error will block subsequent tasks from executing and report the error message so you can debug and fix. 
An additional feature of Jobmon is that if you encounter and error and fix the bug, it's simple to resume the workflow
from the last point of failure, saving the tedium of re-running previous steps that have already completed unnecessarily.

For more details on Jobmon, please refer to the [documentation](https://scicomp-docs.ihme.washington.edu/jobmon/current/).

#### Creating a Local Server

Jobmon requires some infrastructure in order to run, namely a back end web service and a database. At IHME these are hosted
on a private Kubernetes cluster and a reserved database host respectively. However, users without access to the UW VPN 
are unable to access either of these resources, requiring manual instantiation of these structures prior to running. 

`nox` provides a handy way to provide a one-line command that installs custom dependencies and runs the web service. 
To create this local web service, run `pip install nox` if not already installed and subsequently `nox -s setup_local_server`
from the root level of the repository. You should see the app start to run: 

```buildoutcfg
 * Serving Flask app 'jobmon.server.web.app_factory'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:27546
 * Running on http://10.104.145.4:27546
Press CTRL+C to quit
```

Note that this will run the web service in the foreground of your terminal, and block execution of any other commands. 
If you want to free up your terminal, say for running your actual workflow, you can send this process to the background
by pressing `Ctrl + Z` followed by `bg` in your terminal (OSX or Linux only). Any Jobmon workflows created should be 
configured and able to run against this local web service automatically. 

When done with work remember to kill this process to avoid consuming resources unnecessarily on your computer. 
You can do so using `jobs` to retrieve jobs sent to the background, `fg <id>` on the ID of that background job
to bring it back to the foreground, and simply `Ctrl + C` to kill the process. 


### Subsets and Submodels

A key requirement of onemod is the ability to flexibly model different sets of fixed/random effects. To facilitate computation, 
we need to be able to split up the data across different axes arbitrarily. The concept of subsets exists to work nicely with different
chunks of data containing different identifying attributes. 

The `groupby` parameters set in the settings.yml file identify subgroups in the input dataset - e.g. a value of 
`[year_id, sex_id]` indicates that rows are uniquely identified by year and sex and can thus be modeled independently. 

However, data volume is not always evenly distributed across the groups - certain years or locations can contain more data
than other groupings. To enforce smaller groups and thus quicker computation, we can further split up a `submodel` into 
additional `subsets` (without doing this, we'll be waiting unnecessarily for large subsets to complete fitting and might
run into memory issues with those larger subsets). 

## Unit Testing

The `tests/` folder of this repository contain a series of unit tests, at this moment mainly concerned with unit testing
the orchestration layer. This project uses the `pytest` framework for running unit tests. 

You can run the test suite using `nox`, a Python package that manages virtual environments for testing purposes. Nox 
abstracts away a lot of the complexity of building environments; all you need is a conda environment with nox installed,
then you can run tests by calling `nox -s tests` from the root level of this repository. 


## Integration Testing

Unit tests rely on mocking sample datasets and settings files to test that the correct set of tasks is generated. However,
validating results relies on actually running real workflows against representative datasets. 

To test on the cluster:
1. Follow the standard procedure to login and obtain a compute session with srun. 
2. Create a conda environment and install onemod from source. 
   1. Clone this repository, and from the root level run `pip install -e .`
      1. The `-e` flag means you can edit the code in the cloned repository and those changes will immediately propagate, allowing for rapid iteration.
3. Follow the instructions in the main README in this repository to either being new workflows or resume an existing one. 



