.. _setup:

Setting Up Your Development Environment
=======================================

Follow the steps below to set up a local development environment for `OneMod`. You can use either a Python virtual environment (`venv`) or a Conda environment, depending on your preference. The instructions below include both options.

Before starting, ensure you have:

- `Python` installed (see **Python Versions** below for requirements).
- `Make` installed on your system.

---

1. Clone the Repository
------------------------

First, clone the `OneMod` repository and navigate to the project directory:

.. code-block:: bash

   git clone https://github.com/ihmeuw-msca/onemod.git
   cd onemod

---

2. Check Required Python Versions
----------------------------------

`OneMod` requires a specific Python version to ensure compatibility. Refer to the `pyproject.toml` file for supported versions:

.. code-block:: toml

    [project]
    name = "onemod"
    requires-python = ">=3.10, <3.13"

---

Ensure you have a compatible Python version installed, or use `conda` to set up a new environment with the required version:

.. code-block:: bash

    python --version

---

3. Set Up the Development Environment
--------------------------------------

You can choose between a **virtual environment (venv)** or a **Conda environment** to set up your development environment.

**Option 1: Using Virtual Environment (venv)**

Run the following commands to set up the virtual environment:

.. code-block:: bash

   make setup ENV_TYPE=venv PYTHON_VERSION=3.10

This will:
- Create a virtual environment in the `.venv` directory.
- Install the required dependencies (including development tools).

Activate the environment:

.. code-block:: bash

   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

**Option 2: Using Conda**

Run the following commands to set up a Conda environment:

.. code-block:: bash

   make setup ENV_TYPE=conda CONDA_ENV=onemod PYTHON_VERSION=3.10

This will:
- Create a Conda environment named `onemod` (or supply custom name of your choice).
- Install the required dependencies (including development tools).

Activate the Conda environment:

.. code-block:: bash

   conda activate onemod

---

4. Verify the Setup
-------------------

After setting up the environment, verify that everything works as expected:

.. code-block:: bash

   make list-vars

This will display the current values of key environment variables, including `ENV_TYPE`, `PYTHON_VERSION`, and `CONDA_ENV`.

To confirm that `pre-commit` hooks and tools (e.g., `mypy`, `ruff`) are working, run:

.. code-block:: bash

   pre-commit run --all-files

---

5. Start Developing!
--------------------

Congratulations! You’re ready to start contributing to `OneMod`.

To manually run development tools, you can use the `Makefile` shortcuts:

- **Run `mypy` for type checking**:

.. code-block:: bash

   make mypy

---

- **Clean up the environment**:

.. code-block:: bash

   make clean

---

For details on testing, contributing, or other development workflows, see the corresponding sections in the documentation:

- **Testing**: :ref:`running_tests`
- **Contributing Code**: :ref:`contributing_code`
- **Contributing Documentation**: :ref:`contributing_docs`

---

6. Notes for Contributors
-------------------------

- **Python Versions**: Ensure you are using the correct Python version (see `pyproject.toml`).
- **Dependencies**: Dependencies are managed in `pyproject.toml`. Use `pip install -e ".[dev]"` for manual installation if needed.
- **Makefile**: Use the `Makefile` for consistent setup and tooling.
- **Pre-commit Hooks**: Pre-commit hooks (e.g., `mypy`, `ruff`) ensure code quality. They are automatically installed during setup.

---

That’s it! If you encounter any issues during setup, please refer to the project's `CONTRIBUTING.md` or reach out for help.
