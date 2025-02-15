.. _contributing_docs:

=============================
Contributing to Documentation
=============================

To contribute to the documentation:

1. **Install dependencies**:

   Ensure you have the ``docs`` optional dependencies installed:

   .. code-block:: bash

       pip install -e ".[docs]"

2. **Build the documentation locally**:

   Build the HTML documentation to preview changes:

   .. code-block:: bash

       sphinx-build -b html docs docs/_build

   Open ``docs/_build/index.html`` in your browser to view the docs.

   *Note*: If you need to clean up the build artifacts, you can run:

   .. code-block:: bash

       rm -rf docs/_build

3. **Update the relevant `.rst` files**:

   Make your edits in the ``docs/`` folder.

4. **Submit your changes**:

   Follow the :ref:`general contribution workflow <contributing_code>` to open a pull request with your documentation updates.
