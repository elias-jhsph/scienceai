"""Tests for LLM client utilities."""

from unittest.mock import MagicMock


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

        # Should have error message in one of the tool results
        # The result is a list of messages, and we need to check each one properly
        found_error = False
        for msg in result:
            content = msg.get("content")
            if content is not None and "ERROR" in str(content):
                found_error = True
                break
        assert found_error, f"Expected ERROR in result messages, got: {result}"


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
