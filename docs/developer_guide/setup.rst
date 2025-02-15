.. _setup:

=======================================
Setting Up Your Development Environment
=======================================

Follow the steps below to set up a local development environment for **OneMod**. You can use either a Python virtual environment (``venv``) or a Conda environment, depending on your preference. The instructions below include both options.

Before starting, ensure you have:

- ``Python`` installed (see **Python Versions** below for requirements).
- ``Make`` installed on your system.

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

Ensure you have a compatible Python version installed, or use ``conda`` to set up a new environment with the required version:

.. code-block:: bash

    python --version

3. Set Up the Development Environment
--------------------------------------

You can choose between a **virtual environment (venv)** or a **Conda environment** to set up your development environment.

.. admonition:: Tip

   You may specify the Python version and environment type as arguments to the ``make setup`` command, or specify them as environment variables. See ``.env.example`` for an example ``.env`` file.

**Option 1: Using Virtual Environment (venv)**

Run the following commands to set up the virtual environment:

.. code-block:: bash

   make setup ENV_TYPE=venv PYTHON_VERSION=3.10

This will:

- Create a virtual environment in the ``.venv`` directory.
- Install the required dependencies (including development tools).

Activate the environment:

.. code-block:: bash

   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

**Option 2: Using Conda**

Run the following commands to set up a Conda environment:

.. code-block:: bash

   make setup ENV_TYPE=conda CONDA_ENV=onemod PYTHON_VERSION=3.10

This will:

- Create a Conda environment named ``onemod`` (or supply custom name of your choice).
- Install the required dependencies (including development tools).

Activate the Conda environment:

.. code-block:: bash

   conda activate onemod

4. Verify the Setup
-------------------

After setting up and **activating your environment**, verify that everything works as expected:

To confirm that ``pre-commit`` hooks and tools (e.g., ``mypy``, ``ruff``) are working, you can run:

.. code-block:: bash

   pre-commit run --all-files


5. Start Developing
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
- **Makefile**: Use the ``Makefile`` for consistent setup and tooling. Be sure to update it if changing setup processes.
- **Pre-commit Hooks**: Pre-commit hooks (e.g., ``mypy``, ``ruff``) ensure code quality. They are automatically installed during setup.

In addition, please see :ref:`Contributing Code <contributing_code>` for guidelines on contributing to the codebase.

That’s it! If you encounter any issues during setup, please refer to :ref:`OneMod Support <onemod_support>` or reach out for help.
