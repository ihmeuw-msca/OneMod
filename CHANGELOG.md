# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]


[1.0.1] - 2025-02-25

### Added

- Added `deptry` to pyproject.toml to track dependencies.

### Removed

- Removed unused `DataIOHandler` class.
- Removed modeling code (examples, model stages, model configs, utils, tests)

### Fixed

- Pass stage fields as kwargs in `stage.__init__()` to allow default fields in custom stages.

[1.0.0] - 2025-02-14

### Changed

- Package redesigned to allow more flexibility and customization.

[0.2.2] - 2024-03-07

### Fixed

- Sorted data by `age_mid` before plotting rover covariates to correct line order.
