import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_mixed")


def test_mixed_thinking_sequence():
    """
    Test if we can enable thinking effectively when the previous assistant message
    (which called a tool) did NOT have thinking blocks.
    """
    print("\n--- TEST: Thinking after Non-Thinking Tool Call ---")

    # Mock configuration
    # NOTE: You need a valid API key in env or this will fail authentication,
    # but we are testing the Request Validation (Bad Request) mostly.
    # If we get 401, it means the request SHAPE was valid.
    # If we get 400 with "thinking... something", then it was invalid.

    # Construct a history where:
    # 1. User: "Do X"
    # 2. Assistant: Tool Use (NO THINKING)
    # 3. User: Tool Result

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

    # We want to enable thinking for the NEXT response
    # provider = AnthropicProvider(LLMConfig())

    # We will mock the client.messages.create to just print arguments or
    # use a real call if keys are available.
    # Actually, the quickest way to verify logic "inside" the provider (the safety check)
    # is to instantiate the provider and call _chat_completion_api, but that makes a real network call.

    # Let's inspect the `check_thinking_safety` logic by running a modified version of it locally
    # mirroring the current code in llm_providers.py (BEFORE my tool choice fix, effectively).

    converted_messages = messages  # Simplify: assume already converted for this test checking logic
    # request_args = {"thinking": {"type": "enabled", "budget_tokens": 1024}}

    # --- LOGIC UNDER TEST (Reflecting current llm_providers.py) ---
    if converted_messages:
        last_assistant_idx = -1
        for i in range(len(converted_messages) - 1, -1, -1):
            if converted_messages[i]["role"] == "assistant":  # type: ignore
                last_assistant_idx = i
                break

        if last_assistant_idx != -1:
            last_msg = converted_messages[last_assistant_idx]
            content = last_msg.get("content", "")  # type: ignore
            has_thinking = False

            if isinstance(content, list) and len(content) > 0:
                for block in content:
                    if isinstance(block, dict) and (
                        block.get("type") == "thinking" or block.get("type") == "redacted_thinking"
                    ):
                        has_thinking = True
                        break

            is_tool_result_seq = False
            if converted_messages[-1]["role"] == "user" and isinstance(converted_messages[-1]["content"], list):  # type: ignore
                for block in converted_messages[-1]["content"]:  # type: ignore
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        is_tool_result_seq = True
                        break

            print(f"Is Tool Result Seq: {is_tool_result_seq}")
            print(f"Has Thinking in Prev Assistant: {has_thinking}")

            if is_tool_result_seq and not has_thinking:
                print(">> LOGIC OUTCOME: would DISABLE thinking")
            else:
                print(">> LOGIC OUTCOME: would KEEP thinking")


if __name__ == "__main__":
    test_mixed_thinking_sequence()
