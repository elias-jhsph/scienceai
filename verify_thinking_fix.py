import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify")


def process_messages(converted_messages, reasoning_effort="medium"):
    # This logic matches the UPDATED scienceai/llm_providers.py
    request_args = {}

    # Simulate adding thinking parameter
    if reasoning_effort:
        request_args["thinking"] = {"type": "enabled", "budget_tokens": 2048}

        if converted_messages:
            # Find the last assistant message
            last_assistant_idx = -1
            for i in range(len(converted_messages) - 1, -1, -1):
                if converted_messages[i]["role"] == "assistant":
                    last_assistant_idx = i
                    break

            # Check condition: if we are continuing a tool conversation sequence
            if last_assistant_idx != -1:
                last_msg = converted_messages[last_assistant_idx]
                content = last_msg.get("content", "")
                has_thinking = False

                if isinstance(content, list) and len(content) > 0:
                    # UPDATED CHECK: Check all blocks
                    for block in content:
                        if isinstance(block, dict) and (
                            block.get("type") == "thinking" or block.get("type") == "redacted_thinking"
                        ):
                            has_thinking = True
                            break

                # If we are in a sequence where the assistant previously called a tool, and we are sending the result back
                is_tool_result_seq = False
                if converted_messages[-1]["role"] == "user" and isinstance(converted_messages[-1].get("content"), list):
                    # check if it's a tool result
                    for block in converted_messages[-1]["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            is_tool_result_seq = True
                            break

                logger.info(f"Last Assistant Index: {last_assistant_idx}")
                logger.info(f"Has Thinking: {has_thinking}")
                logger.info(f"Is Tool Result Sequence: {is_tool_result_seq}")

                if is_tool_result_seq and not has_thinking:
                    logger.warning("Disabling thinking!")
                    request_args.pop("thinking", None)

    # FINAL SAFETY: If thinking is disabled (either not requested or disabled by safety check),
    # we MUST strip any thinking blocks from the history to avoid 400 InvalidRequestError.
    if "thinking" not in request_args and converted_messages:
        for msg in converted_messages:
            if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
                # Filter out thinking blocks
                new_content = [
                    block
                    for block in msg["content"]
                    if isinstance(block, dict) and block.get("type") not in ("thinking", "redacted_thinking")
                ]
                msg["content"] = new_content

    return request_args, converted_messages


# Test case 1: Thinking was missing, should disable thinking AND strip (though nothing to strip if missing)
print("\n--- TEST CASE 1: Missing thinking block -> Disable + Strip ---")
messages_1 = [
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
args_1, msgs_1 = process_messages(messages_1)
print(f"Thinking Enabled: {'thinking' in args_1}")
print(f"Messages Content: {msgs_1[1]['content']}")  # Should match origin (no thinking)

# Test case 2: Has thinking block, should KEEP thinking and KEEP block
print("\n--- TEST CASE 2: Has thinking block -> Keep Enabled ---")
messages_2 = [
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
args_2, msgs_2 = process_messages(messages_2)
print(f"Thinking Enabled: {'thinking' in args_2}")
print(f"Messages Content len: {len(msgs_2[1]['content'])}")  # Should be 2 (thinking + tool)

# Test case 3: Forcible disable (simulate user turned it off), should STRIP thinking block
print("\n--- TEST CASE 3: Forcible disable -> Strip Thinking ---")
messages_3 = [
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
# Simulate force disable by passing reasoning_effort=None
args_3, msgs_3 = process_messages(messages_3, reasoning_effort=None)
print(f"Thinking Enabled: {'thinking' in args_3}")
print(f"Messages Content len: {len(msgs_3[1]['content'])}")  # Should be 1 (ONLY tool use)
print(f"Messages Content types: {[b['type'] for b in msgs_3[1]['content']]}")

# Test case 4: "False Negative" scenario - detection fails (hypothetically) so we disable,
# BUT we must assert that the stripping logic works and prevents the 400 error.
# We simulate detection failure by mocking `has_thinking=False` inside the function,
# but here we can just create a scenario where thinking is present but we disable it.
# (Basically same as Test 3).
