# Contributing to OneMod

Thank you for your interest in contributing to **OneMod**! We welcome contributions of all kinds, including bug reports, feature suggestions, documentation improvements, and code contributions. This guide outlines how to get involved.

For a more detailed guide on the development workflow, see the [Contributing Code](https://ihmeuw-msca.github.io/OneMod/1.0.0/developer_guide/contributing_code.html) section of our documentation.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Style Guides](#style-guides)
- [Need Help?](#need-help)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](https://github.com/ihmeuw-msca/OneMod/blob/main/CODE_OF_CONDUCT.md). Please ensure that all interactions remain respectful and constructive.

## Pull Request Process

We use GitHub pull requests (PRs) for all code contributions. To submit a PR:

1. **Fork the repository** and create a feature branch. See [Contributing Code](https://ihmeuw-msca.github.io/OneMod/1.0.0/developer_guide/contributing_code.html) for more details.
2. **Write clean, well-documented code** and ensure all changes are covered by tests.
3. **Run linters and type checks** (`mypy`, `ruff`, etc.).
4. **Run the test suite** (`pytest`) and ensure all tests pass.
5. **Update `CHANGELOG.md`** if your change introduces new features or fixes.
6. **Open a PR** against the `main` branch with a clear title and description.
7. **Follow up on reviews**—maintainers may request changes or ask for clarification.

All PRs trigger automated tests via GitHub Actions. Your PR will be reviewed once checks pass.

## Reporting Bugs

If you find a bug, please help us improve OneMod by reporting it:

- **Review the documentation and [OneMod Support](https://ihmeuw-msca.github.io/OneMod/1.0.0/getting_started/onemod_support.html) page** before opening a new issue.
- **Check existing issues** to avoid duplicates. If the issue is already reported, add any additional information you have, and/or upvote the issue.
- **Open a new issue**, providing:
  - A clear title and description.
  - Steps to reproduce the bug.
  - Expected vs. actual behavior.
  - System information (Python version, OS, dependencies).
  - Any relevant logs or screenshots.

You can report issues in our [GitHub Issues](https://github.com/ihmeuw-msca/OneMod/issues).

## Suggesting Enhancements

We welcome feature requests and ideas for improvement! To suggest an enhancement:

- **Check existing issues** to see if a similar idea has been discussed.
- **Create a new issue** describing:
  - The problem your suggestion addresses.
  - A proposed solution or approach.
  - Any relevant use cases.

Discussions may take place before implementation to refine the proposal.

## Style Guides

### Code Style

Consistency is the name of the game. OneMod follows standard Python style conventions. Before submitting code:

- **Lint with Ruff**: `ruff --check`
- **Type-check with MyPy**: `mypy src/ tests/`
- **Format code with Ruff (auto-fix)**: `ruff --fix`

See [Contributing Code](https://ihmeuw-msca.github.io/OneMod/1.0.0/developer_guide/contributing_code.html) for details.

### Commit Messages

- Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) (e.g., `feat: add new validation rule`).
- Keep messages clear and descriptive.
- Use present-tense, imperative-style writing.

## Need Help?

If you have any questions, feel free to ask by:

- Opening a discussion in the [GitHub Discussions](https://github.com/ihmeuw-msca/OneMod/discussions) tab.
- Joining an issue or PR discussion.

Please review the [OneMod Support](https://ihmeuw-msca.github.io/OneMod/1.0.0/getting_started/onemod_support.html) page for more resources. We appreciate your contributions—thank you for helping improve OneMod!
