.. _branches_and_versioning:

=======================
Branches and Versioning
=======================

**OneMod** uses a standard branching strategy and versioning scheme to manage development and releases. This page outlines the key aspects of this strategy.

Release Cycles
================

**OneMod** development is organized into release cycles. Each release cycle culminates in a new major or minor version of the package. The release cycle consists of the following stages:

1. **Development**: New features and improvements are contributed via feature (``feature/`` or ``feat/``) and bugfix (``bugfix/``) branches.
2. **Pull Request**: These branches are to be merged into the ``release/x.y`` branch after they are ready and pull requests are approved.
3. **Release**: After testing, the ``release/x.y`` branch is merged into the ``main`` branch, and a new version is released.

Out-of-Sync Releases
====================

In some cases, it may be necessary to release a new version of the package that is not in sync with the current development branch. This can happen when a critical bug is found in the ``main`` branch and needs to be fixed immediately. In this case, a hotfix (``hotfix/``) branch is created from the ``main`` branch, and the fix is applied there. Once the fix is complete and tested, the hotfix branch is merged back into the ``main`` branch and a new *patch* version (``x.y.z``) is released.

When this happens, we need to ensure that the ``release/x.y`` branch is also updated to keep in sync with the latest changes to ``main``. Due to our intentional branch protection rules, we cannot merge the ``main`` branch into the ``release/x.y`` branch directly. Instead, once a new hotfix or patch is merged into ``main`` and a new release is created, we do the following:

1. Locally, ensure that the ``release/x.y`` branch is up to date:

    .. code-block:: bash

        git checkout release/x.y
        git pull origin release/x.y

2. Merge the ``main`` branch into the ``release/x.y`` branch:

    .. code-block:: bash

        git merge main

    If there are any conflicts, resolve them and commit the changes:

    .. code-block:: bash

        git add .
        git commit -m "Resolve merge conflicts from main into release/x.y"

3. Push to a temporary branch:

    .. code-block:: bash

        git checkout -b merge-main-into-release-x.y
        git push origin merge-main-into-release-x.y

4. Open a pull request to merge the temporary branch into the ``release/x.y`` branch.

    - Give it a clear title, e.g., "Merge hotfix 1.0.1 into release/1.1".
    - Add a description of the changes, pointing out any merge conflicts that needed to be resolved.

5. Once the pull request is approved and merged, the ``release/x.y`` branch will be up to date with the latest changes from the ``main`` branch. The temporary branch can then be deleted.

Branches
========

**OneMod** uses the following branch conventions:

- **main**: The main branch contains the latest stable release of the package. This is the branch that users should install from.
- **release/x.y**: These branches are used for maintaining specific versions of the package. Each release branch corresponds to a specific version of the package and is used for collecting features and bug fixes.
- **feature/** (or **feat/**): Feature branches are used for developing new features. These branches are created from the ``release/x.y`` branch and are merged back into it when the feature is complete.
- **bugfix/**: Bugfix branches are used for fixing bugs in the package. These branches are created from the ``release/x.y`` branch and are merged back into it when the bug is fixed.
- **hotfix/**: Hotfix branches are used for urgent fixes that need to be applied to the ``main`` branch immediately. These branches are created from the ``main`` branch and are merged back into it when the fix is complete.
