# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.6] - 2025-12-27

### Fixed
- **RECITATION/Safety Block Handling**: Fixed crashes in PDF processing when the API returns `None` content due to RECITATION or safety blocks. Now gracefully handles empty responses with warnings.
- **Figure Description Null Safety**: Added null check for figure description API responses to prevent crashes during page processing.

### Improved
- **Favicon**: Added a 🧪 emoji favicon to the menu page for better browser tab identification.

## [0.4.5] - 2025-12-27

### Fixed
- **Google AI Rate Limiting**: Added automatic retry with exponential backoff for 429/RESOURCE_EXHAUSTED errors, with up to 3 retries and parsed delay times.
- **Google AI Token Counting**: Fixed `count_tokens` to only use `system_instruction` on Vertex AI path, avoiding API errors on the API key path.
- **Google AI Schema Compatibility**: Added stripping of unsupported `format` and `default` fields from tool parameter schemas to prevent API errors.

## [0.4.4] - 2025-12-27

### Fixed
- **FileNotFoundError**: Fixed a critical crash where the `io` directory was missing from the installed package and not created at runtime.
- **Runtime Directory Creation**: Added automatic creation of the `io` directory at startup to ensure robustness across different installation environments.

### Internal
- **Packaging Consistency**: Updated `MANIFEST.in` and `pyproject.toml` to ensure all static assets, templates, prompts, and the `io` directory are correctly included in the distribution.
- **Version Tracking**: Added a `.gitkeep` file to the `io` directory to ensure it is tracked by version control.

## [0.4.3] - 2025-12-13

### Improved
- **Terminology Consistency**: Updated UI labels and prompts from "Data Collection" to "Data Extraction" for clearer communication.
- **Progress Indicator**: Enhanced with two-phase transformation—linear lag (0-50%) then exponential catch-up (50-100%) for smoother, more natural progress bar animation.
- **Pause Functionality**: Enhanced pause/resume workflow with automatic modal closure and button state reset on completion.
- **Agent Prompts**: Enhanced Analyst and PI prompts with critical rules about tool call execution, error recovery, delegation workflow, and completion requirements.
- **Close Dialog**: Improved messaging and formatting for project close confirmation dialog.
- **Parallel Calls Persistence**: Parallel analyst replicate setting (1×, 2×, 3×) now persists across page loads and sessions.
- **Gemini Thinking Support**: Added support for preserving `thought_signature` and `thinking` blocks in bundle validator for Gemini 3 Pro with thinking capabilities.

### Fixed
- **Author Extraction**: Fixed metadata author parsing with improved error handling and fallback logic for missing or malformed author data.
- **Context Percentage**: Added null checks to prevent errors when `context_percentage` is `None` in chat templates.
- **Dark Mode**: Fixed chat input container background transparency issue in dark theme.
- **Database Error Handling**: Added automatic error status clearing before reprocessing papers to prevent stale error states.
- **Database Locking**: Added `lock_project()` method for safe thread termination by acquiring write locks on critical database files.
- **File Path Fix**: Fixed file path handling in PI generated content to ensure correct paths are returned.
- **Test Fixes**: Fixed token clamping and counting tests to properly mock Anthropic beta API and Google Vertex client.

### Internal
- **Code Cleanup**: Removed obsolete test files (`verify_thinking_fix.py`, `verify_thinking_mixed.py`) and cleaned up whitespace/formatting across codebase.
- **Prompt Cleanup**: Removed minimal `analyst_prepend_openai.txt` file (OpenAI models follow base prompt directly).

## [0.4.2] - 2025-12-13

### Added
- **Parallel Analyst Delegation**: PI can now spawn multiple independent analyst instances in parallel for replicated research. User-controllable via UI dropdown (1×, 2×, 3× replicates).
- **New Data Type**: Added `text_block_list` for extracting multiple narrative sections or related text segments.
- **Local Config Support**: Config files now support a priority-based lookup—`scienceai-config.json` in the current working directory takes precedence over the global `~/Documents/ScienceAI/scienceai-config.json`.

### Improved
- **Context Compression**: Enhanced compression with metadata extraction—now differentiates between `delegate_research` and `run_python_code` calls, extracting analyst names, collection names, and referenced files for better context preservation.
- **WebSocket Stability**: Graceful handling of `BrokenPipeError` exceptions in all WebSocket handlers (`/discussion`, `/papers`, `/progress`) to prevent page refresh loops.
- **Fallback Reload Timeout**: Increased from 5 seconds to 30 seconds to reduce unnecessary page reloads during normal operation.
- **UI Icons**: Migrated all FontAwesome icons to native emojis (👁️, 📥, ↩️, 🗑️, ➕) for better cross-platform compatibility and reduced external dependencies.
- **Undo Enhancement**: Undo now also removes parallel analyst replicates (e.g., "Analyst Name copy 1", "copy 2", etc.).
- **Compression Lock**: Added guard to prevent multiple simultaneous compression operations.
- **PI Code Execution**: Improved documentation for relative file paths in `run_python_code` tool.

### Fixed
- **Orphaned Tool Calls**: Added healing for orphaned tool calls during context compression.
- **Thread Safety**: Added locking around analyst list mutations to prevent race conditions during parallel delegation.

### Internal
- **Model Updates**: Default OpenAI model updated from `gpt-5.1` to `gpt-5.2`.
- **OpenAI API**: Moved `max_completion_tokens` and `reasoning_effort` to `extra_body` for better API compatibility.
- **Provider Status**: Fixed `get_current_provider_name()` to use `load_config()` instead of cached config.

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

[0.4.6]: https://github.com/elias-jhsph/scienceai/compare/0.4.5...0.4.6
[0.4.5]: https://github.com/elias-jhsph/scienceai/compare/0.4.4...0.4.5
[0.4.4]: https://github.com/elias-jhsph/scienceai/compare/0.4.3...0.4.4
[0.4.3]: https://github.com/elias-jhsph/scienceai/compare/0.4.2...0.4.3
[0.4.2]: https://github.com/elias-jhsph/scienceai/compare/0.4.1...0.4.2
[0.4.1]: https://github.com/elias-jhsph/scienceai/compare/0.3.1...0.4.1
[0.3.1]: https://github.com/elias-jhsph/scienceai/releases/tag/0.3.1
