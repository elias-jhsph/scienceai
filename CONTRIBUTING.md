# Contributing to ScienceAI

Thank you for your interest in contributing to ScienceAI! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/scienceai.git
   cd scienceai
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/eliaswestonfarber/scienceai.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- An OpenAI API key (for integration tests)
- Tesseract OCR (for PDF text extraction)

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Set up your API key** (for integration tests):
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Run linting
ruff check src/scienceai

# Run type checking
mypy src/scienceai
```

## Making Changes

### Branching Strategy

1. **Create a feature branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Keep your branch up to date**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Commit Messages

Write clear, concise commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests when relevant

**Good examples:**
```
Add support for PDF batch processing

Fix memory leak in data extraction pipeline

Update documentation for CLI usage
```

## Code Style

We use automated tools to maintain consistent code style:

### Formatting and Linting

```bash
# Format code
ruff format src/scienceai tests/

# Check for linting issues
ruff check src/scienceai tests/

# Auto-fix linting issues
ruff check --fix src/scienceai tests/
```

### Type Hints

- Add type hints to all new functions and methods
- Use `from __future__ import annotations` for modern type syntax
- Run `mypy` to check for type errors:
  ```bash
  mypy src/scienceai
  ```

### Documentation

- Add docstrings to all public functions, classes, and modules
- Use Google-style docstrings:
  ```python
  def function_name(param1: str, param2: int) -> bool:
      """Short description of the function.

      Longer description if needed, explaining the purpose
      and behavior of the function.

      Args:
          param1: Description of param1.
          param2: Description of param2.

      Returns:
          Description of the return value.

      Raises:
          ValueError: When param1 is empty.
      """
  ```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_database_manager.py -v

# Run with coverage
pytest tests/ --cov=scienceai --cov-report=html

# Run only fast tests (skip slow/integration)
pytest tests/ -m "not slow and not integration"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with the `test_` prefix
- Name test functions with the `test_` prefix
- Use descriptive test names that explain what is being tested
- Use pytest fixtures for common setup

**Example:**
```python
import pytest
from scienceai.database_manager import sha256sum

class TestSHA256Sum:
    """Tests for the sha256sum function."""

    def test_returns_valid_hex_string(self, temp_dir):
        """sha256sum should return a 64-character hex string."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("hello")

        result = sha256sum(str(test_file))

        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_content_produces_same_hash(self, temp_dir):
        """Identical content should produce identical hashes."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        content = "test content"

        file1.write_text(content)
        file2.write_text(content)

        assert sha256sum(str(file1)) == sha256sum(str(file2))
```

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v
   ```

2. **Ensure code quality checks pass**:
   ```bash
   ruff check src/scienceai tests/
   mypy src/scienceai
   ```

3. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub with:
   - A clear title describing the change
   - A description explaining:
     - What the change does
     - Why it's needed
     - How it was tested
   - Reference any related issues

### Pull Request Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- List of specific changes made

## Testing
- How the changes were tested
- Any new tests added

## Related Issues
Fixes #123
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - Python version
   - Operating system
   - ScienceAI version
6. **Logs/Error Messages**: Any relevant error output

### Feature Requests

When requesting features, please include:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives Considered**: Other approaches you've thought about
4. **Additional Context**: Any other relevant information

## Questions?

If you have questions about contributing, feel free to:
- Open a GitHub issue
- Start a discussion on the repository

Thank you for contributing to ScienceAI! 🎉
