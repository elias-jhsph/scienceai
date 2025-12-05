"""Tests for LLM client utilities."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestGetApiKey:
    """Tests for API key retrieval."""

    def test_returns_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return API key from OPENAI_API_KEY environment variable."""
        test_key = "sk-test-env-key-12345"
        monkeypatch.setenv("OPENAI_API_KEY", test_key)

        # Import after setting env var to avoid module-level initialization
        from scienceai.llm import _get_api_key

        result = _get_api_key()
        assert result == test_key

    def test_returns_key_from_config_file(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
        """Should return API key from config file when env var not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Create mock config file
        config_dir = temp_dir / "Documents" / "ScienceAI"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "scienceai-keys.json"
        config_file.write_text(json.dumps({"openai": "sk-test-file-key"}))

        # Mock expanduser to return our temp directory
        monkeypatch.setattr(os.path, "expanduser", lambda x: str(temp_dir))

        from scienceai.llm import _get_api_key

        result = _get_api_key()
        assert result == "sk-test-file-key"

    def test_raises_error_when_no_key_found_non_interactive(
        self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path
    ) -> None:
        """Should raise ValueError when no API key is available."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(os.path, "expanduser", lambda x: str(temp_dir))

        # Mock stdin.isatty to return False (non-interactive)
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        from scienceai.llm import _get_api_key

        with pytest.raises(ValueError, match="OpenAI API key not found"):
            _get_api_key()


class TestTrimHistory:
    """Tests for history trimming functionality."""

    def test_returns_history_under_limit(self) -> None:
        """Should return history unchanged when under token limit."""
        from scienceai.llm import trim_history

        history = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = trim_history(history.copy(), token_limit=10000)
        assert len(result) == 3

    def test_removes_messages_when_over_limit(self) -> None:
        """Should remove messages (preserving system) when over token limit."""
        from scienceai.llm import trim_history

        history = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "First user message " * 100},
            {"role": "assistant", "content": "First response " * 100},
            {"role": "user", "content": "Second user message " * 100},
            {"role": "assistant", "content": "Second response " * 100},
        ]

        result = trim_history(history.copy(), token_limit=500)

        # System message should be preserved (index 0)
        assert result[0]["role"] == "system"
        # Should have fewer messages
        assert len(result) < 5


class TestUseToolsSync:
    """Tests for synchronous tool execution."""

    def test_executes_function_successfully(self) -> None:
        """Should execute tool function and return results."""
        from scienceai.llm import use_tools_sync

        def test_func(arg1: str) -> str:
            return f"Result: {arg1}"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].message.tool_calls[0].function.name = "test_func"
        mock_response.choices[0].message.tool_calls[0].function.arguments = '{"arg1": "test"}'
        mock_response.choices[0].message.tool_calls[0].id = "call_123"
        mock_response.choices[0].message.content = "Calling test function"

        arguments = {"tools": [{"function": {"name": "test_func"}}]}

        result = use_tools_sync(
            mock_response,
            arguments,
            function_dict={"test_func": test_func},
        )

        # Should have assistant message and tool result
        assert any(msg.get("role") == "assistant" for msg in result)
        assert any(msg.get("role") == "tool" and "Result: test" in msg.get("content", "") for msg in result)

    def test_handles_function_not_found(self) -> None:
        """Should return error when function is not in function_dict."""
        from scienceai.llm import use_tools_sync

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].message.tool_calls[0].function.name = "unknown_func"
        mock_response.choices[0].message.tool_calls[0].function.arguments = "{}"
        mock_response.choices[0].message.tool_calls[0].id = "call_456"
        mock_response.choices[0].message.content = None

        result = use_tools_sync(
            mock_response,
            {"tools": []},
            function_dict={},
        )

        # Should have error message
        assert any("ERROR" in msg.get("content", "") for msg in result)


class TestUpdateStopEvent:
    """Tests for stop event management."""

    def test_updates_global_stop_event(self) -> None:
        """Should update the global STOP_EVENT."""
        from threading import Event

        from scienceai.llm import update_stop_event

        new_event = Event()
        update_stop_event(new_event)

        from scienceai.llm import STOP_EVENT as updated_event  # noqa: N811

        assert updated_event is new_event

    def test_can_set_to_none(self) -> None:
        """Should be able to set STOP_EVENT to None."""
        from scienceai.llm import update_stop_event

        update_stop_event(None)

        from scienceai.llm import STOP_EVENT

        assert STOP_EVENT is None
