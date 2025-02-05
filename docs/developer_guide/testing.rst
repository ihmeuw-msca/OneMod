.. _running_tests:

=============
Running Tests
=============

Quickstart
==========

**OneMod** uses ``pytest`` for testing. To ensure all tests pass, follow these steps:

1. Run the full test suite:

   .. code-block:: bash

       pytest

2. Generate a test coverage report (optional):

   .. code-block:: bash

       pytest --cov=onemod

3. View detailed coverage in HTML format (optional):

   .. code-block:: bash

       pytest --cov=onemod --cov-report=html

   Open the ``htmlcov/index.html`` file in your browser to inspect test coverage.

Remember to write tests for new features or bug fixes!

Organizing Tests
================

- **Unit Tests**: Located in the ``tests/unit/`` directory. These tests focus on individual components or functions.
- **Integration Tests**: Located in the ``tests/integration/`` directory. These tests verify the interaction between multiple components.
- **End-to-End Tests**: Located in the ``tests/e2e/`` directory. These tests simulate real-world scenarios to ensure the entire system works as expected.
- **Helpers**: Located in the ``tests/helpers/`` directory. These are utility functions and classes used by the **OneMod** test suite.

Marking Tests
=============

``pytest`` provides a convenient system for marking tests in order to categorize them or skip them under certain conditions. For example, to mark a test to be skipped if a certain condition is met:

.. code-block:: python

    import pytest

    @pytest.mark.skipif(condition, reason="Condition not met")
    def test_conditional():
        # Test code here
        pass

Or, to mark a test as "requires_jobmon":

.. code-block:: python

    import pytest

    @pytest.mark.requires_jobmon
    def test_requires_jobmon_function():
        # Test code here
        pass

Within **OneMod**, we make use of the following custom markers:

- **unit**: Marks unit tests.
- **integration**: Marks integration tests.
- **e2e**: Marks end-to-end tests.
- **requires_data**: Marks tests that require specific data files to be present which are not included in version control. These tests are generally meant for testing the data loading and processing functionality of **OneMod**.
- **requires_jobmon**: Marks tests that require the jobmon package to be installed and are meant to be run specifically using jobmon. These tests are generally meant for testing the jobmon integration and are not included in the GitHub Actions CI workflow.

.. admonition:: Tip

    Test paths and markers are specified in ``pytest.ini``. If you add a new marker, remember to update this file accordingly, as well as the relevant documentation (i.e. this section).

To run tests with specific markers, use the ``-m`` option:

.. code-block:: bash

    pytest -m "marker_name"

For example, to run all tests marked as "unit":

.. code-block:: bash

    pytest -m "unit"

To run all tests except those marked as "requires_jobmon":

.. code-block:: bash

    pytest -m "not requires_jobmon"

For more information on pytest markers, see the `pytest documentation <https://docs.pytest.org/en/stable/example/markers.html>`_.
