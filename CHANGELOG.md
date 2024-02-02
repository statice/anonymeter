# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2024-02-02

### Changed

- numba is updated to 0.58 to allow for the newer numpy version
- numpy version range is adapted accordingly to numba's requirements
- python 3.11 is allowed
- pandas version is relaxed to allow for pandas >= 2
  * added additional CI pipeline for pandas 2

### Fixed

- singling out evaluators getting stuck on multivariate queries

## [0.0.2] - 2023-07-10

### Added

- CNIL mention (#18)
- Customized logging on module level (#19)

### Fixed

- Pre-commit errors (#19)


## [0.0.1] - 2023-04-24

### Added

- Initial release
