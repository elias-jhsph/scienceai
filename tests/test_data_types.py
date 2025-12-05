"""Tests for data type validation and schema generation."""

import json
from pathlib import Path

import pytest


class TestDataTypesConfiguration:
    """Test data types JSON configuration files."""

    @pytest.fixture
    def data_types_path(self) -> Path:
        """Return path to data_types.json."""
        return Path(__file__).parent.parent / "src" / "scienceai" / "data_types.json"

    @pytest.fixture
    def data_types_docs_path(self) -> Path:
        """Return path to data_types_docs.json."""
        return Path(__file__).parent.parent / "src" / "scienceai" / "data_types_docs.json"

    def test_data_types_file_exists(self, data_types_path: Path) -> None:
        """Verify data_types.json exists."""
        assert data_types_path.exists(), "data_types.json should exist"

    def test_data_types_is_valid_json(self, data_types_path: Path) -> None:
        """Verify data_types.json is valid JSON."""
        with open(data_types_path) as f:
            data = json.load(f)
        assert isinstance(data, dict), "data_types.json should be a dictionary"

    def test_data_types_docs_exists(self, data_types_docs_path: Path) -> None:
        """Verify data_types_docs.json exists."""
        assert data_types_docs_path.exists(), "data_types_docs.json should exist"

    def test_data_types_docs_is_valid_json(self, data_types_docs_path: Path) -> None:
        """Verify data_types_docs.json is valid JSON."""
        with open(data_types_docs_path) as f:
            data = json.load(f)
        assert isinstance(data, dict), "data_types_docs.json should be a dictionary"

    def test_data_types_has_required_types(self, data_types_path: Path) -> None:
        """Verify data_types.json contains expected data types."""
        with open(data_types_path) as f:
            data = json.load(f)

        expected_types = ["number", "text_block", "categorical_value", "boolean_value", "date"]
        for dtype in expected_types:
            assert dtype in data, f"data_types.json should contain '{dtype}' type"
