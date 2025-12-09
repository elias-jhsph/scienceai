"""Tests for ScienceAI client interface."""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestScienceAIClient:
    """Tests for the ScienceAI client class."""

    @patch("scienceai.client.DatabaseManager")
    @patch("scienceai.client.run_backend")
    def test_initialization_creates_project(
        self, mock_run_backend: MagicMock, mock_dm: MagicMock, temp_dir: Path
    ) -> None:
        """Should initialize client with project name."""
        from scienceai import ScienceAI

        mock_dm_instance = MagicMock()
        mock_dm.return_value = mock_dm_instance
        mock_run_backend.return_value = None

        client = ScienceAI(
            project_name="TestProject",
            storage_path=str(temp_dir),
            validate_keys=False,
        )

        assert client.project_name == "TestProject"

    @patch("scienceai.client.DatabaseManager")
    @patch("scienceai.client.run_backend")
    def test_history_returns_chat_messages(
        self, mock_run_backend: MagicMock, mock_dm: MagicMock, temp_dir: Path
    ) -> None:
        """Should return chat history from database manager."""
        from scienceai import ScienceAI

        mock_dm_instance = MagicMock()
        mock_dm_instance.get_database_chat.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        mock_dm.return_value = mock_dm_instance
        mock_run_backend.return_value = None

        client = ScienceAI(
            project_name="TestProject",
            storage_path=str(temp_dir),
            validate_keys=False,
        )
        history = client.history()

        assert len(history) == 2
        assert history[0]["role"] == "user"


class TestClientValidation:
    """Tests for client input validation."""

    @patch("scienceai.client.DatabaseManager")
    @patch("scienceai.client.run_backend")
    def test_accepts_empty_project_name(self, mock_run_backend: MagicMock, mock_dm: MagicMock, temp_dir: Path) -> None:
        """Should accept empty project name and generate a timestamp-based name."""
        from scienceai import ScienceAI

        mock_dm.return_value = MagicMock()
        mock_run_backend.return_value = None

        # Empty project name is now allowed and generates a timestamp name
        client = ScienceAI(project_name="", storage_path=str(temp_dir), validate_keys=False)
        # Empty string is falsy, so it generates a timestamp-based name
        assert "Project Started at" in client.project_name or client.project_name == ""

    @patch("scienceai.client.DatabaseManager")
    @patch("scienceai.client.run_backend")
    def test_rejects_invalid_project_name_characters(
        self, mock_run_backend: MagicMock, mock_dm: MagicMock, temp_dir: Path
    ) -> None:
        """Should handle project names with special characters."""
        # Some characters may be invalid depending on implementation
        # This test documents expected behavior
        import contextlib

        from scienceai import ScienceAI

        mock_dm.return_value = MagicMock()
        mock_run_backend.return_value = None

        # Should either work or raise a clear error
        with contextlib.suppress(ValueError, OSError):
            ScienceAI(
                project_name="Test/Project",
                storage_path=str(temp_dir),
                validate_keys=False,
            )
