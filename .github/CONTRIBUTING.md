# Contributing to OneMod

Thank you for your interest in contributing to **OneMod**! We welcome contributions of all kinds, including bug reports, feature suggestions, documentation improvements, and code contributions. This guide outlines how to get involved.

For a more detailed guide on the development workflow, see the [Contributing Code](https://ihmeuw-msca.github.io/OneMod/1.1.0/developer_guide/contributing_code.html) section of our documentation.

## Table of Contents

- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Style Guides](#style-guides)
- [Need Help?](#need-help)

## Pull Request Process

We use GitHub pull requests (PRs) for all code contributions. To submit a PR:

1. **Fork the repository** and create a feature branch. See [Contributing Code](https://ihmeuw-msca.github.io/OneMod/1.1.0/developer_guide/contributing_code.html) for more details.
2. **Write clean, well-documented code** and ensure all changes are covered by tests.
3. **Run linters and type checks** (`mypy`, `ruff`, etc.).
4. **Run the test suite** (`pytest`) and ensure all tests pass.
5. **Update `CHANGELOG.md`** if your change introduces new features or fixes.
6. **Open a PR** against the `main` branch with a clear title and description. We recommend using the [Pull Request Template](https://github.com/ihmeuw-msca/OneMod/blob/main/.github/PULL_REQUEST_TEMPLATE.md) to provide context for your changes:
   - **Title**: Use a descriptive title that summarizes the changes.
   - **Description**: Explain what your PR does, why it's necessary, and how it has been tested.
   - **Types of Changes**: Check the appropriate boxes for the types of changes your PR introduces (bug fix, new feature, etc.).
   - **Checklist**: Go through the checklist to ensure all requirements are met.
7. **Follow up on reviews**—maintainers may request changes or ask for clarification.

All PRs to `main` trigger automated tests via GitHub Actions. Your PR will be reviewed once checks pass.

## Reporting Bugs

If you find a bug, please help us improve OneMod by reporting it:

- **Review the documentation and [OneMod Support](https://ihmeuw-msca.github.io/OneMod/1.1.0/getting_started/onemod_support.html) page** before opening a new issue.
- **Check existing issues** to avoid duplicates. If the issue is already reported, add any additional information you have, and/or upvote the issue.
- **Open a new issue** with a clear title and description. We recommend using the [Bug Report Issue Template](https://github.com/ihmeuw-msca/OneMod/blob/main/.github/ISSUE_TEMPLATE/bug-report-issue.md) as a starting point to provide context:
  - Steps to reproduce the bug.
  - Expected vs. actual behavior.
  - System information (Python version, OS, dependencies).
  - Any relevant logs or screenshots.

You can report issues in our [GitHub Issues](https://github.com/ihmeuw-msca/OneMod/issues).

## Suggesting Enhancements

We welcome feature requests and ideas for improvement! To suggest an enhancement:

- **Check existing issues** to see if a similar idea has been proposed.
- **Create a new issue** describing:
  - The problem your suggestion addresses.
  - A proposed solution or approach.
  - Any relevant use cases.

Conversations may take place before implementation to refine the proposal.

## Style Guides

### Code Style

Consistency is the name of the game. OneMod follows standard Python style conventions. Before submitting code:

- **Lint with Ruff**: `ruff --check`
- **Type-check with MyPy**: `mypy src/ tests/`
- **Format code with Ruff (auto-fix)**: `ruff --fix`

See [Contributing Code](https://ihmeuw-msca.github.io/OneMod/1.1.0/developer_guide/contributing_code.html) for details.

### Commit Messages

- Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) (e.g., `feat: add new validation rule`).
- Keep messages clear and descriptive.
- Use present-tense, imperative-style writing.

## Need Help?

Please review the [OneMod Support](https://ihmeuw-msca.github.io/OneMod/1.1.0/getting_started/onemod_support.html) page for more resources. We appreciate your contributions—thank you for helping improve OneMod!
