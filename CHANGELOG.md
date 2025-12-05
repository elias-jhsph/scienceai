# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with pytest
- Type hints throughout the codebase
- Pre-commit hooks for code quality
- Ruff linting and formatting configuration
- MyPy type checking configuration
- `py.typed` marker for PEP 561 compliance
- CHANGELOG.md for tracking changes
- CONTRIBUTING.md with development guidelines
- Improved CI/CD pipeline with matrix testing

### Changed
- Updated all GitHub Actions to latest versions (v4/v5)
- Enhanced `.gitignore` with comprehensive patterns
- Improved `pyproject.toml` with modern configuration
- Added proper public API exports in `__init__.py`
- Refactored `llm.py` with type hints and better error handling

## [0.3.3] - 2024-12-05

### Added
- Bundle validator for data extraction quality assurance
- Context compression feature for long conversations
- Progress tracking WebSocket for real-time updates
- Context usage indicator in the UI

### Changed
- Improved analyst workflow with better error handling
- Enhanced data extraction with multiple extraction modes

## [0.3.2] - 2024-11-XX

### Added
- Python client interface (`ScienceAI` class)
- Programmatic access to all features
- Background processing with async operations

### Fixed
- Various bug fixes and stability improvements

## [0.3.1] - 2024-10-XX

### Added
- Initial public release
- Principal Investigator (PI) agent system
- Analyst agent creation and management
- PDF processing and text extraction
- Structured data extraction with JSON schemas
- Web UI for project management
- CSV and JSON export functionality
- Checkpoint and project save/restore

### Features
- Multi-agent architecture for research tasks
- Automatic paper metadata detection
- Provenance tracking for extracted data
- Interactive chat interface with "Show work" transparency

[Unreleased]: https://github.com/elias-jhsph/scienceai/compare/0.3.3...HEAD
[0.3.3]: https://github.com/elias-jhsph/scienceai/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/elias-jhsph/scienceai/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/elias-jhsph/scienceai/releases/tag/0.3.1
