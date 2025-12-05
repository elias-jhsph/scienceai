"""Pytest configuration and shared fixtures for ScienceAI tests."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_pdf_path() -> Path:
    """Return path to the sample test PDF if it exists."""
    pdf_path = Path(__file__).parent.parent / "test_paper.pdf"
    if not pdf_path.exists():
        pytest.skip("test_paper.pdf not found")
    return pdf_path


@pytest.fixture
def mock_openai_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a mock OpenAI API key for testing."""
    test_key = "sk-test-key-for-testing-only"
    monkeypatch.setenv("OPENAI_API_KEY", test_key)
    return test_key


@pytest.fixture
def project_storage(temp_dir: Path) -> Path:
    """Create a temporary storage directory for project data."""
    storage = temp_dir / "scienceai_test_storage"
    storage.mkdir(parents=True, exist_ok=True)
    return storage
