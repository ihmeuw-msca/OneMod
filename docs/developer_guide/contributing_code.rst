.. _contributing_code:

=================
Contributing Code
=================

Contribution Workflow
---------------------

Follow these steps to contribute code to **OneMod**:

1. **Fork the repository (optional for core contributors)**:

   Fork the **OneMod** repository on GitHub to your account.

2. **Create a branch**:

   Work on your feature or fix in a new branch:

   .. code-block:: bash

       git checkout -b feat/my-feature

   See :ref:`Branches and Versioning <branches_and_versioning>` for more information on branch naming conventions for the **OneMod** project.

   .. admonition:: Tip

        While not strictly enforced, we recommend following best practices when contributing commits. See `Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`_ for conventions to follow when naming your branch and writing commit messages.

3. **Make your changes**:

   Write clean, well-documented code. Add tests to verify your changes.

   See :ref:`Contributing to Documentation <contributing_docs>` for documentation guidelines and :ref:`Running Tests <running_tests>` for testing guidelines.

   If making many iterative updates, it is also recommended to commit often, which will trigger pre-commit checks. If not, at least type check your code as you go to catch any issues early:

   .. code-block:: bash

      mypy src/ tests/

   You may also want to manually run Ruff to check for any linting issues:

   .. code-block:: bash

      ruff --check

4. **Run tests**:

   Ensure all tests pass before pushing your code:

   .. code-block:: bash

      pytest

   It is often useful to take advantage of our pytest markers to run only specific tests during iterative development. See the :ref:`Running Tests <running_tests>` section for more details.

5. **Update CHANGELOG.md**:

   If your changes include new features or bug fixes, update the repository's ``CHANGELOG.md`` file to reflect these changes.

   Example ``CHANGELOG.md`` entries:

   .. code-block:: markdown

      ## [Unreleased]

      ### Added

      - Added ability to ... Description of the new feature.

      ### Changed

      - Changed behavior of ... Description of the change.

      ### Fixed

      - Fixed bug where ... Description of the bug fix.

6. **Submit a pull request**:

   Push your branch to your forked repository and submit a pull request. Creating a pull request should automatically trigger a GitHub Actions workflow to ensure your code passes all tests and checks.

   If all tests and checks pass, you will see a green check mark on the builds page in your PR.

Keeping Your Fork Updated
-------------------------

To ensure your fork stays up to date with the latest changes from the **OneMod** repository, follow these steps:

1. **Add the Upstream Repository (Only Needed Once)**:

   If you haven’t already added the main repository as an upstream remote, do so:

   .. code-block:: bash

      git remote add upstream https://github.com/ihmeuw-msca/OneMod.git

   Verify that ``upstream`` is correctly set up:

   .. code-block:: bash

      git remote -v

   You should see ``upstream`` pointing to the main repository.

2. **Fetch the Latest Changes from Upstream**:

   Before updating, ensure you're on the ``main`` branch:

   .. code-block:: bash

      git checkout main

   Then, fetch the latest changes:

   .. code-block:: bash

      git fetch upstream

3. **Sync Your Fork’s main Branch**:

   Merge upstream changes into your ``main`` branch:

   .. code-block:: bash

      git merge upstream/main

   Push the updated ``main`` to your fork:

   .. code-block:: bash

      git push origin main

4. **Keep Your Feature Branch Updated**:

   If you're working on a feature branch, update it as well:

   .. code-block:: bash

      git checkout feat/my-feature
      git merge main
