.. _setup:

=======================================
Setting Up Your Development Environment
=======================================

Follow the steps below to set up a local development environment for **OneMod**. You can use either a Python virtual environment (``venv``) or a Conda environment, depending on your preference. The instructions below include both options.

1. Clone the Repository
------------------------

First, clone the **OneMod** repository and navigate to the project directory:

.. code-block:: bash

   git clone https://github.com/ihmeuw-msca/onemod.git
   cd onemod

2. Check Required Python Versions
----------------------------------

**OneMod** requires a specific Python version to ensure compatibility. Refer to the ``pyproject.toml`` file for supported versions:

.. code-block:: toml

   [project]
   name = "onemod"
   requires-python = ">=3.10, <3.13"

3. Set Up a Virtual or Conda Environment
-----------------------------------------

You can choose between a **virtual environment (venv)** or a **Conda environment** to set up your development environment.

**Option 1: Using Virtual Environment (venv)**

   1. Ensure you have a compatible Python version installed:

      .. code-block:: bash

         python --version

   2. Create a virtual environment in the ``.venv`` directory:

      .. code-block:: bash

         python -m venv .venv

   3. Activate the virtual environment:

      .. code-block:: bash

         source .venv/bin/activate  # On Windows, use .venv\Scripts\activate

**Option 2: Using Conda**

   1. Create a Conda environment named ``onemod`` (or name it as you prefer):

      .. code-block:: bash

         conda create -n onemod python=3.10  # Substitute with desired Python version

   2.  Activate the Conda environment:

      .. code-block:: bash

         conda activate onemod

**(Optional) Ensure pip is up to date:**

   .. code-block:: bash

      python -m pip install --upgrade pip

4. Install required dependencies
---------------------------------

**Example 1:** Install only the required dependencies for local development and testing:

   .. code-block:: bash

      pip install -e ".[dev]"

**Example 2:** If you will be working on ``jobmon``-related tasks, or want to test ``jobmon`` execution functionality on the **Slurm Cluster**, you will also need to install the ``jobmon`` dependencies:

   .. code-block:: bash

      pip install -e ".[dev, jobmon]"

**Example 3:** If you will be contributing to the documentation (or simply wish to build the docs locally), you will need the ``docs`` dependencies as well:

   .. code-block:: bash

      pip install -e ".[dev, docs]"

.. admonition:: Tip

   The most common setup for internal contributors working on the **Slurm Cluster** is to install all dependencies at once:

   .. code-block:: bash

      pip install -e ".[dev, jobmon, docs]"

5. Install the pre-commit git hooks
------------------------------------

Finally, to ensure code quality and consistency, install the pre-commit hooks:

   .. code-block:: bash

      pre-commit install

6. Verify the Setup
-------------------

After setting up and **activating your environment**, verify that everything works as expected:

To confirm that ``pre-commit`` hooks and tools (e.g., ``mypy``, ``ruff``) are working, you can run:

.. code-block:: bash

   pre-commit run --all-files


7. Start Developing
-------------------

You should be ready to start contributing to **OneMod**!

To manually run development tools, first ensure your environment is activated, for example:

.. code-block:: bash

   source .venv/bin/activate  # Or `conda activate onemod`


Then, you can run the following commands as needed:

- **Run `pytest` for testing**:

.. code-block:: bash

   pytest


- **Run `mypy` for type checking**:

.. code-block:: bash

   mypy src/ tests/


- **Run `ruff` for linting**:

.. code-block:: bash

   ruff --check


For details on testing, contributing, or other development workflows, see the corresponding sections in the documentation:

- :ref:`Running Tests <running_tests>`
- :ref:`Contributing Code <contributing_code>`
- :ref:`Contributing to Documentation <contributing_docs>`


Notes for Contributors
----------------------

- **Python Versions**: Ensure you are using the correct Python version (see ``pyproject.toml``).
- **Dependencies**: Dependencies are managed in ``pyproject.toml``. Use ``pip install -e ".[dev]"`` for manual installation if needed. Please update the ``pyproject.toml`` file if you add new dependencies.
- **Pre-commit Hooks**: Pre-commit hooks (e.g., ``mypy``, ``ruff``) ensure code quality. They are automatically installed during setup.

In addition, please see :ref:`Contributing Code <contributing_code>` for guidelines on contributing to the codebase.

That’s it! If you encounter any issues during setup, please refer to :ref:`OneMod Support <onemod_support>` or reach out for help.
