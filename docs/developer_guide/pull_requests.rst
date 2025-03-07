.. _pull_requests:

=============
Pull Requests
=============

.. admonition:: Note

    Before proceeding, please review our `GitHub CONTRIBUTING.md <https://github.com/ihmeuw-msca/OneMod/blob/main/.github/CONTRIBUTING.md>`_ for the most up-to-date guidelines for contributing to the **OneMod** codebase.

Guidelines for Submitting a Pull Request
----------------------------------------

Before submitting a Pull Request (PR), please check that it meets the following criteria.

These guidelines apply to **all OneMod PRs**, regardless of the target branch:

1. **Code Quality:**
    - Includes tests for any new features or bug fixes.
    - Follows the coding standards outlined in the :ref:`Contributing Code <contributing_code>` section. This includes passing linting checks and type checking.
    - Passes all tests and checks in the GitHub Actions workflow.

2. **Documentation:**
    - Updates to the documentation are included in the PR.
    - Documentation changes are clear and concise.
    - Follows the guidelines in the :ref:`Contributing to Documentation <contributing_docs>` section.

3. **Pull Request Description:**
    - Indicates the type of change (bug fix, new feature, etc.).
    - Provides a clear and concise description of the changes made and why they are necessary.
    - References any related issues.
    - Describes how the changes have been tested, if applicable.

    .. admonition:: Tip

        For external contributors, please use the `GitHub PR template <https://github.com/ihmeuw-msca/OneMod/blob/main/.github/PULL_REQUEST_TEMPLATE.md>`_ to ensure you include all necessary information.

PRs targeting the ``main`` branch should adhere to stricter guidelines, as they represent changes that will immediately go into the next release if merged:

    1. Ensure that the ``CHANGELOG.md`` file is updated to include an entry describing the change, bumping the **version number**, and adding the **date of the release** if applicable.
    2. If applicable, update the version number in:
        - ``pyproject.toml``
        - ``docs/meta.toml``
        - Any other relevant files.
    3. Any **breaking changes** and **major new features** should be clearly documented and communicated to users. **Determine whether announcements and/or deprecation warnings will be necessary**, and communicate this to the team.
    4. For external contributor, large or significant changes should be **discussed in an issue** before submission.
    5. PRs directly targeting ``main`` must follow the `GitHub PR template <https://github.com/ihmeuw-msca/OneMod/blob/main/.github/PULL_REQUEST_TEMPLATE.md>`_. This template includes a checklist to ensure that all necessary steps have been taken before merging.

Guidelines for Reviewing Pull requests
--------------------------------------

Once a PR is submitted, it will be reviewed by the maintainers of the **OneMod** repository assuming **all GitHub Actions checks pass**.

Reviewers of **all OneMod PRs** should follow the following guidelines:

    1. Ensure that all existing **CI checks and tests** pass.
    2. Review whether there are **changes to the API** and determine if changes are backward compatible. Ensure that the destination branch is appropriate for the changes.
    3. Ensure the code follows our **coding standards** outlined in the :ref:`Contributing Code <contributing_code>` section. Check for any potential bugs or issues. Ask the author to clarify any parts of the code that are unclear.
    4. **Review test coverage**, ensure that tests are included for any new features or bug fixes.
    5. **Ensure that documentation has been updated** in accordance with any code changes, especially if the changes affect the API.
    6. For new features and bug fixes, **check that CHANGELOG.md has been updated** accordingly.

In addition, reviewers of PRs to the ``main`` branch should:

    1. Ensure that the ``CHANGELOG.md`` file has been updated with a new entry for the release, complete with **version number and date**.
    2. Ensure that the version number in ``pyproject.toml`` has been updated to the new version number.
    3. Ensure that the version number in ``docs/meta.toml`` has been updated to the new version number.
    4. Ensure that any other references to the old version number, for instance links to the live documentation, have been updated to the new version number.
    5. Name a **release manager** to oversee the release process, who will generate the release notes and tag the release in GitHub.
