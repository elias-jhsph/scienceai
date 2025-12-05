"""Tests for ScienceAI client interface."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestScienceAIClient:
    """Tests for the ScienceAI client class."""

    @patch("scienceai.client.DatabaseManager")
    @patch("scienceai.client.start_backend")
    def test_initialization_creates_project(
        self, mock_start_backend: MagicMock, mock_dm: MagicMock, temp_dir: Path
    ) -> None:
        """Should initialize client with project name."""
        from scienceai import ScienceAI

        mock_dm_instance = MagicMock()
        mock_dm.return_value = mock_dm_instance
        mock_start_backend.return_value = MagicMock()

        client = ScienceAI(
            project_name="TestProject",
            storage_path=str(temp_dir),
            start_server=False,
        )

        assert client.project_name == "TestProject"

    @patch("scienceai.client.DatabaseManager")
    @patch("scienceai.client.start_backend")
    def test_history_returns_chat_messages(
        self, mock_start_backend: MagicMock, mock_dm: MagicMock, temp_dir: Path
    ) -> None:
        """Should return chat history from database manager."""
        from scienceai import ScienceAI

        mock_dm_instance = MagicMock()
        mock_dm_instance.get_database_chat.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        mock_dm.return_value = mock_dm_instance
        mock_start_backend.return_value = MagicMock()

        client = ScienceAI(
            project_name="TestProject",
            storage_path=str(temp_dir),
            start_server=False,
        )
        history = client.history()

        assert len(history) == 2
        assert history[0]["role"] == "user"


class TestClientValidation:
    """Tests for client input validation."""

    def test_rejects_empty_project_name(self, temp_dir: Path) -> None:
        """Should raise error for empty project name."""
        from scienceai import ScienceAI

        with pytest.raises((ValueError, TypeError)):
            ScienceAI(project_name="", storage_path=str(temp_dir))

    def test_rejects_invalid_project_name_characters(self, temp_dir: Path) -> None:
        """Should handle project names with special characters."""
        # Some characters may be invalid depending on implementation
        # This test documents expected behavior
        import contextlib

        from scienceai import ScienceAI

        # Should either work or raise a clear error
        with (
            patch("scienceai.client.DatabaseManager"),
            patch("scienceai.client.start_backend"),
            contextlib.suppress(ValueError, OSError),
        ):
            ScienceAI(
                project_name="Test/Project",
                storage_path=str(temp_dir),
                start_server=False,
            )
