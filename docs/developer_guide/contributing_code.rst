Contributing Code
=================

Follow these steps to contribute code to `OneMod`:

1. **Fork the repository (optional for core contributors)**:
   Fork the `OneMod` repository on GitHub to your account.

2. **Create a branch**:
   Work on your feature or fix in a new branch:

   .. code-block:: bash

       git checkout -b my-feature

3. **Make your changes**:
   Write clean, well-documented code. Add tests to verify your changes.

4. **Run tests**:
   Ensure all tests pass before pushing your code:

   .. code-block:: bash

       pytest

5. **Submit a pull request**:
   Push your branch to your forked repository and submit a pull request.

Keeping Your Fork Updated
-------------------------
For external contributors, keep your fork updated with the main repository:

1. Fetch the latest changes from upstream:

   .. code-block:: bash

       git fetch upstream

2. Merge or rebase the changes into your branch:

   .. code-block:: bash

       git merge upstream/main  # Or rebase: git rebase upstream/main
