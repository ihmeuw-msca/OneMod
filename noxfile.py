import nox
from nox.sessions import Session

python = "3.10"

@nox.session(python=python, venv_backend="conda")
def lint(session: Session) -> None:
    """Lint code using various plugins.

    flake8 - a Python library that wraps PyFlakes, pycodestyle and McCabe script.
    flake8-import-order - checks the ordering of your imports.
    flake8-annotations -is a plugin for Flake8 that detects the absence of PEP 3107-style
    function annotations and PEP 484-style type comments.
    flake8-black - "The Uncompromising Code Formatter", https://pypi.org/project/flake8-black/
    """

    args = session.posargs or ["src"]
    # TODO - Add in flake8-docstrings (extension for flake8 which uses pydocstyle to check docstrings.)
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-import-order",
        "flake8-black"
    )
    session.run(
        "flake8",
        *args
    )

@nox.session(python=python, venv_backend="conda")
def black(session):
    args = session.posargs or ["src"]
    session.install("black")
    session.run(
        "black",
        *args
    )

@nox.session(python=python, venv_backend="conda")
def tests(session: Session) -> None:
    """Run the test suite and coverage report."""
    session.install("pytest", "pytest-xdist", "pytest-cov")
    session.install("-e", ".")

    args = session.posargs or ["tests"]

    session.run(
        "pytest",
        "--cov=onemod",
        "--cov-report=term",
        *args
    )

@nox.session(python=python, venv_backend="conda")
def typecheck(session: Session) -> None:
    """Type check code."""
    args = session.posargs or ["src"]
    session.install("mypy", "types-PyYAML", "pandas-stubs")
    session.install("-e", ".")
    session.run("mypy", "--explicit-package-bases", *args)