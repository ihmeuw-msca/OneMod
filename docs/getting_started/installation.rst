Installation
============

The **OneMod** package currently supports Python 3.10, 3.11, and 3.12.
For a list of required dependencies, see
`project.toml <https://github.com/ihmeuw-msca/OneMod/blob/main/pyproject.toml>`_.

* To install from PyPI:

  .. code-block:: bash

     pip install onemod

* To install from GitHub:

  .. code-block:: bash

     pip install git+https://github.com/ihmeuw-msca/OneMod.git

  See `VCS Support <https://pip.pypa.io/en/stable/topics/vcs-support/#git>`_
  for more details, including how to specify a branch name, commit hash,
  or tag name.

* To install with optional dependencies:

  .. code-block:: bash

     pip install onemod[jobmon]

  For a list of optional dependencies, see
  `project.toml <https://github.com/ihmeuw-msca/OneMod/blob/main/pyproject.toml>`_.
