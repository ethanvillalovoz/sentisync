# Changelog

## Unreleased

### Added

- Security policy and code of conduct for public issue and pull request handling.

### Changed

- Contributor setup now defaults to API runtime dependencies.
- Manual deployment no longer injects AWS credentials into the running Flask container.

## v1.0.0 - Public Project Baseline

Initial polished public baseline for SentiSync.

### Added

- Research and portfolio-ready README with architecture, setup, API, Docker, DVC, and deployment documentation.
- Backend smoke tests for health, prediction validation, model/vectorizer wiring, chart validation, and preprocessing behavior.
- GitHub Actions CI for Python compile checks, unit tests, and Docker build verification.
- Manual AWS deployment path through GitHub Actions.
- Changelog and contributor templates tailored to the project.

### Changed

- Canonicalized Python dependency names.
- Updated Docker build to cache dependency layers and install required NLTK corpora.
- Replaced stale container names and placeholder CI commands with SentiSync-specific automation.
- Hardened Flask startup defaults and NLTK fallback behavior without changing model inference semantics.
