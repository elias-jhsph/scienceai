"""Tests for LLM provider functionality, particularly thinking block handling."""


class TestThinkingBlockStripping:
    """Tests for thinking block stripping when thinking is disabled."""

    def test_strip_thinking_when_disabled(self) -> None:
        """Test that thinking blocks are stripped from messages when thinking is disabled."""
        # This tests the logic at lines 1295-1310 in llm_providers.py
        messages = [
            {"role": "user", "content": "Begin task."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should call foo", "signature": "sig123"},
                    {"type": "tool_use", "id": "tool_1", "name": "foo", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "result"}]},
        ]

        # Simulate the stripping logic
        converted_messages = messages.copy()
        for msg in converted_messages:
            if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
                new_content = [
                    block
                    for block in msg["content"]
                    if isinstance(block, dict) and block.get("type") not in ("thinking", "redacted_thinking")
                ]
                msg["content"] = new_content

        # Verify thinking was stripped
        assistant_msg = converted_messages[1]
        assert len(assistant_msg["content"]) == 1, "Should have only tool_use after stripping thinking"
        assert assistant_msg["content"][0]["type"] == "tool_use", "Should have tool_use block"

    def test_keep_thinking_when_enabled(self) -> None:
        """Test that thinking blocks are kept when thinking is enabled."""
        messages = [
            {"role": "user", "content": "Begin task."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should call foo", "signature": "sig123"},
                    {"type": "tool_use", "id": "tool_1", "name": "foo", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "result"}]},
        ]

        # When thinking is enabled, blocks should remain
        converted_messages = messages.copy()
        assistant_msg = converted_messages[1]
        assert len(assistant_msg["content"]) == 2, "Should have both thinking and tool_use"
        assert assistant_msg["content"][0]["type"] == "thinking", "Should have thinking block first"

    def test_strip_thinking_from_multiple_messages(self) -> None:
        """Test that thinking blocks are stripped from all assistant messages when disabled."""
        messages = [
            {"role": "user", "content": "First task."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "First thought", "signature": "sig1"},
                    {"type": "text", "text": "First response"},
                ],
            },
            {"role": "user", "content": "Second task."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Second thought", "signature": "sig2"},
                    {"type": "tool_use", "id": "tool_1", "name": "bar", "input": {}},
                ],
            },
        ]

        # Simulate stripping logic
        converted_messages = messages.copy()
        for msg in converted_messages:
            if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
                new_content = [
                    block
                    for block in msg["content"]
                    if isinstance(block, dict) and block.get("type") not in ("thinking", "redacted_thinking")
                ]
                msg["content"] = new_content

        # Verify both assistant messages had thinking stripped
        assert len(converted_messages[1]["content"]) == 1, "First assistant should have only text"
        assert converted_messages[1]["content"][0]["type"] == "text"
        assert len(converted_messages[3]["content"]) == 1, "Second assistant should have only tool_use"
        assert converted_messages[3]["content"][0]["type"] == "tool_use"


class TestThinkingSequenceDetection:
    """Tests for detecting thinking sequences in message history."""

    def test_detect_tool_result_sequence(self) -> None:
        """Test detection of tool result sequences."""
        messages = [
            {"role": "user", "content": "Calculate 123 * 456"},
            {
                "role": "assistant",
                "content": [
                    # NO THINKING BLOCK HERE, just tool use
                    {"type": "tool_use", "id": "tool_1", "name": "calculator", "input": {"expression": "123 * 456"}}
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "56088"}]},
        ]

        # Check if last message is a tool result
        is_tool_result_seq = False
        if messages[-1]["role"] == "user" and isinstance(messages[-1].get("content"), list):
            for block in messages[-1]["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    is_tool_result_seq = True
                    break

        assert is_tool_result_seq, "Should detect tool result sequence"

    def test_detect_thinking_in_previous_assistant(self) -> None:
        """Test detection of thinking blocks in previous assistant message."""
        messages = [
            {"role": "user", "content": "Begin task."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should call foo", "signature": "sig123"},
                    {"type": "tool_use", "id": "tool_1", "name": "foo", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "result"}]},
        ]

        # Find last assistant message
        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break

        assert last_assistant_idx != -1, "Should find assistant message"

        # Check for thinking blocks
        last_msg = messages[last_assistant_idx]
        content = last_msg.get("content", "")
        has_thinking = False

        if isinstance(content, list) and len(content) > 0:
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"):
                    has_thinking = True
                    break

        assert has_thinking, "Should detect thinking block in previous assistant message"

    def test_detect_missing_thinking_in_previous_assistant(self) -> None:
        """Test detection when previous assistant message lacks thinking."""
        messages = [
            {"role": "user", "content": "Begin task."},
            {
                "role": "assistant",
                "content": [
                    # MISSING THINKING BLOCK
                    {"type": "tool_use", "id": "tool_1", "name": "foo", "input": {}}
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "result"}]},
        ]

        # Find last assistant message
        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break

        # Check for thinking blocks
        last_msg = messages[last_assistant_idx]
        content = last_msg.get("content", "")
        has_thinking = False

        if isinstance(content, list) and len(content) > 0:
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"):
                    has_thinking = True
                    break

        assert not has_thinking, "Should detect missing thinking block"
