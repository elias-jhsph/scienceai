# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2025-12-09

### Added
- **Multi-LLM Provider Support**:
  - **OpenAI**: Direct API support.
  - **Anthropic**: Dual-route support via Direct API (API Key) or Google Vertex AI (GCP Service Account).
  - **Google Gemini**: Dual-route support via Google AI Studio (API Key) or Vertex AI (GCP Service Account).
  - **Flexible Authentication**: Unified config handling for API keys and Service Account paths.
- **Automated Quality Assurance**:
  - **Pre-commit Hooks**: Integrated linting (Ruff), formatting, type checking (MyPy), and security scanning (Bandit) running automatically on commit.
  - **CI/CD Pipeline**: Enhanced GitHub Actions workflows for automated testing (`test.yml`) and PyPI publishing.
- **Reset Conversation**: Feature to clear history and conversation state, including a fix for database lock issues.
- **Add Papers**: New functionality to upload additional PDFs to an existing project via the main menu.
- **Anthropic Thinking Blocks**: Support for "thinking" blocks in Claude 3.7 / 4.5 responses for improved reasoning transparency.

### Improved
- **UI/UX Enhancements**:
  - Dark Mode support for JSON viewer.
  - Standardized typography for chat timestamps.
  - **Context Awareness**: Fixed visibility of context limit warnings and improved token clamping/compression logic.
  - **Visual Memory Indicator**: Added a floating "Brain" emoji 🧠 indicator to visualize current context window usage.
- **Agent Reliability & Stability**:
  - "Null Content" reminder for Analyst agents to prevent empty responses.
  - Disabled parallel tool execution for Gemini/Vertex agents to prevent race conditions.
  - Improved system prompts for more robust data collection and schema adherence.
  - Self-healing for tool use mismatches and orphaned tool calls in the PI.
  - Fixed schema generation errors (missing keys, invalid property names).

### Internal
- **Refactoring & Maintenance**:
  - Removed `create_arbitrary_csv` tool (superseded by Python code execution).
  - Decoupled `principal_investigator.py` from direct `dictdatabase` dependencies.
  - **System-Wide Multi-Model Refactor**: Comprehensive refactor of `process_paper.py`, `backend.py`, `llm.py`, and agent prompts to support provider-agnostic logic, replacing all hardcoded OpenAI defaults with configurable provider settings.
  - Migrated ad-hoc test scripts to a standard `tests/` directory.
  - Added `py.typed` marker for better type support.
- **Packaging**: Fixed missing `io` directory in package distribution (`pyproject.toml`).

## [0.3.1] - 2025-11-25

### Architecture & Core
- **Principal Investigator Refactor**: Major expansion of `principal_investigator.py` (390+ lines added) introducing `verify_completeness` and improved delegation logic.
- **Data Extraction**: Significant audit of `data_extractor.py` (490+ lines added) adding robust schema validation and reflection capabilities.
- **Analyst Agents**: Enhanced `analyst.py` (300+ lines added) with better paper selection and schema generation.
- **Async Backend**: Rewrote `backend.py` to support asynchronous operations for better concurrency.

### Frontend & Visualization
- **UI Overhaul**: Complete redesign of `apps.css` (1200+ lines changed) and templates (`app.html`, `papers.html`) for a modern, responsive interface.
- **Visualization**: Added `db_element.html` and updated `jquery.json-viewer` for better data inspection.

### Testing & Stability
- **Verification Scripts**: Added dedicated verification scripts (`verify_error_handling.py`, `verify_ingestion.py`, `verify_load_data.py`) which may later evolve into the test suite.
- **Debugging Tools**: Added `debugging_scripts/` directory with test cases for O1 models and data extraction.

### Added
- Initial public release
- Principal Investigator (PI) agent system
- Multi-agent architecture for research tasks
- PDF processing and automatic metadata detection
- Structured data extraction with JSON schemas

[0.4.1]: https://github.com/elias-jhsph/scienceai/compare/0.3.1...0.4.1
[0.3.1]: https://github.com/elias-jhsph/scienceai/releases/tag/0.3.1
