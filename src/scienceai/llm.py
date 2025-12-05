"""LLM client utilities for OpenAI API interactions.

This module provides the OpenAI client initialization and tool calling utilities
for both synchronous and asynchronous operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import tiktoken
from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from threading import Event

    from openai.types.chat import ChatCompletion

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases for clarity
MessageDict = dict[str, Any]
ToolCallDict = dict[str, Any]
FunctionDict = dict[str, Callable[..., Any]]

# Global stop event for graceful shutdown
STOP_EVENT: Event | None = None


def _get_api_key() -> str:
    """Get OpenAI API key from environment or configuration file.

    Returns:
        The OpenAI API key.

    Raises:
        ValueError: If no API key is found and not running interactively.
    """
    # First try environment variable
    if api_key := os.environ.get("OPENAI_API_KEY"):
        return api_key

    # Fall back to config file
    base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
    target_key = os.path.join(base_key_path, "scienceai-keys.json")

    if os.path.exists(target_key):
        try:
            with open(target_key) as file:
                key_list = json.load(file)
            if openai_key := key_list.get("openai"):
                return openai_key
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read API key from config file: {e}")

    # Check if running interactively before prompting
    import sys

    if sys.stdin.isatty():
        try:
            new_key = input("Please enter OpenAI key: ")
            if new_key.strip():
                os.makedirs(os.path.dirname(target_key), exist_ok=True)
                with open(target_key, "w") as file:
                    json.dump({"openai": new_key}, file)
                return new_key
        except (EOFError, KeyboardInterrupt):
            pass

    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
        "or add it to ~/Documents/ScienceAI/scienceai-keys.json"
    )


# Initialize clients
_api_key = _get_api_key()

# Async client for ingestion pipeline
async_client = AsyncOpenAI(api_key=_api_key)

# Sync client for agents and data extraction
client = OpenAI(api_key=_api_key)

# Token encoder for context management
enc = tiktoken.encoding_for_model("gpt-4")


def update_stop_event(stop_event: Event | None) -> None:
    """Update the global stop event for graceful shutdown.

    Args:
        stop_event: Threading event to signal shutdown.
    """
    global STOP_EVENT
    STOP_EVENT = stop_event


def trim_history(history: list[MessageDict], token_limit: int) -> list[MessageDict]:
    """Trim conversation history to fit within token limit.

    Removes messages from the beginning (after system message) to reduce token count.

    Args:
        history: List of message dictionaries.
        token_limit: Maximum number of tokens allowed.

    Returns:
        Trimmed history list.
    """
    for _ in range(len(history)):
        if len(enc.encode(str(history))) > token_limit:
            history.pop(1)  # Keep system message at index 0
        else:
            return history
    return history


async def use_tools(
    chat_response: ChatCompletion | dict[str, Any],
    arguments: dict[str, Any],
    function_dict: FunctionDict | None = None,
    call_functions: bool = True,
    pre_tool_call: bool = False,
) -> list[MessageDict]:
    """Process and execute tool calls from a chat response (async version).

    Args:
        chat_response: OpenAI chat completion response or dict with tool_calls.
        arguments: Original arguments dict containing tool schemas.
        function_dict: Mapping of function names to callable functions.
        call_functions: Whether to actually execute the functions.
        pre_tool_call: If True, return early with just the assistant message.

    Returns:
        List of message dictionaries for conversation history.
    """
    if function_dict is None:
        function_dict = {}

    if isinstance(chat_response, dict):
        tool_calls = chat_response["tool_calls"]
        content = chat_response["content"]
    else:
        tool_calls = chat_response.choices[0].message.tool_calls
        content = chat_response.choices[0].message.content

    tools = arguments.get("tools", [])
    tool_calls_list: list[ToolCallDict] = []

    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_calls_list.append(
                    {
                        "function": {
                            "strict": True,
                            "arguments": tool_call["function"]["arguments"],
                            "name": tool_call["function"]["name"],
                        },
                        "id": tool_call["id"],
                        "type": "function",
                    }
                )
            else:
                tool_calls_list.append(
                    {
                        "function": {
                            "arguments": tool_call.function.arguments,
                            "name": tool_call.function.name,
                        },
                        "id": tool_call.id,
                        "type": "function",
                    }
                )

    # Build assistant message with or without tool calls
    if call_functions:
        if tool_calls_list:
            new_history: list[MessageDict] = [{"content": content, "role": "assistant", "tool_calls": tool_calls_list}]
        else:
            new_history = [{"content": content, "role": "assistant"}]

    if pre_tool_call:
        return new_history

    # If just extracting parameters without execution
    if not call_functions:
        valid_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls_list:
            function_name = tool_call["function"]["name"]
            try:
                parsed_args = json.loads(tool_call["function"]["arguments"])
                valid_calls.append({"name": function_name, "parameters": parsed_args})
            except json.JSONDecodeError:
                pass  # Skip invalid JSON
        return valid_calls

    # Execute tools concurrently
    tasks = []
    for tool_call in tool_calls_list:
        function_name = tool_call["function"]["name"]
        tool_schema = next((t for t in tools if t["function"]["name"] == function_name), None)
        tasks.append(use_tool(tool_call["function"], tool_call["id"], tool_schema, function_dict=function_dict))

    results_and_errors = await asyncio.gather(*tasks)

    tool_results: list[MessageDict] = []
    tool_errors: list[MessageDict] = []

    for res, err in results_and_errors:
        tool_results.extend(res)
        tool_errors.extend(err)

    return new_history + tool_results + tool_errors


async def use_tool(
    tool_call: dict[str, Any],
    tool_id: str,
    tool_schema: dict[str, Any] | None,
    function_dict: FunctionDict | None = None,
) -> tuple[list[MessageDict], list[MessageDict]]:
    """Execute a single tool call (async version).

    Args:
        tool_call: Tool call specification with name and arguments.
        tool_id: Unique identifier for this tool call.
        tool_schema: JSON schema for the tool (for error messages).
        function_dict: Mapping of function names to callables.

    Returns:
        Tuple of (results, errors) message lists.
    """
    if function_dict is None:
        function_dict = {}

    function_name = tool_call["name"]
    results: list[MessageDict] = []
    errors: list[MessageDict] = []

    if function_name not in function_dict:
        errors.append({"content": "ERROR", "role": "tool", "name": function_name, "tool_call_id": tool_id})
        errors.append({"content": "Only use a valid function in your function list.", "role": "system"})
        return results, errors

    called_function = function_dict[function_name]

    try:
        parsed_arguments = json.loads(tool_call["arguments"])
        try:
            # Support both async and sync functions
            if asyncio.iscoroutinefunction(called_function):
                result = await called_function(**parsed_arguments)
            else:
                result = called_function(**parsed_arguments)

            results.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": str(result),
                    "tool_call_id": tool_id,
                }
            )
        except Exception as e:
            error_str = (
                f"Error calling {function_name} function with passed arguments "
                f"{parsed_arguments}: {traceback.format_exc()} \n {e}"
            )
            errors.append(
                {
                    "content": error_str,
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": function_name,
                }
            )

    except json.JSONDecodeError:
        required_arguments = tool_schema["function"]["parameters"]["required"] if tool_schema else "unknown"
        if tool_call["arguments"] == "":
            error_content = (
                f"Your function call did not include any arguments. "
                f"Please try again with the correct arguments: {required_arguments}"
            )
        else:
            error_content = "Your function call did not parse as valid JSON. Please try again"
        errors.append({"content": "ERROR", "role": "tool", "name": function_name, "tool_call_id": tool_id})
        errors.append({"content": error_content, "role": "system"})

    return results, errors


def use_tools_sync(
    chat_response: ChatCompletion | dict[str, Any],
    arguments: dict[str, Any],
    function_dict: FunctionDict | None = None,
    call_functions: bool = True,
    pre_tool_call: bool = False,
) -> list[MessageDict]:
    """Process and execute tool calls from a chat response (synchronous version).

    Args:
        chat_response: OpenAI chat completion response or dict with tool_calls.
        arguments: Original arguments dict containing tool schemas.
        function_dict: Mapping of function names to callable functions.
        call_functions: Whether to actually execute the functions.
        pre_tool_call: If True, return early with just the assistant message.

    Returns:
        List of message dictionaries for conversation history.
    """
    if function_dict is None:
        function_dict = {}

    if isinstance(chat_response, dict):
        tool_calls = chat_response["tool_calls"]
        content = chat_response["content"]
    else:
        tool_calls = chat_response.choices[0].message.tool_calls
        content = chat_response.choices[0].message.content

    tools = arguments.get("tools", [])
    tool_calls_list: list[ToolCallDict] = []

    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_calls_list.append(
                    {
                        "function": {
                            "strict": True,
                            "arguments": tool_call["function"]["arguments"],
                            "name": tool_call["function"]["name"],
                        },
                        "id": tool_call["id"],
                        "type": "function",
                    }
                )
            else:
                tool_calls_list.append(
                    {
                        "function": {
                            "arguments": tool_call.function.arguments,
                            "name": tool_call.function.name,
                        },
                        "id": tool_call.id,
                        "type": "function",
                    }
                )

    if call_functions:
        if tool_calls_list:
            new_history: list[MessageDict] = [{"content": content, "role": "assistant", "tool_calls": tool_calls_list}]
        else:
            new_history = [{"content": content, "role": "assistant"}]

    if pre_tool_call:
        return new_history

    if not call_functions:
        valid_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls_list:
            function_name = tool_call["function"]["name"]
            try:
                arguments_parsed = json.loads(tool_call["function"]["arguments"])
                valid_calls.append({"name": function_name, "parameters": arguments_parsed})
            except json.JSONDecodeError:
                pass
        return valid_calls

    # Execute tools synchronously
    results_and_errors = []
    for tool_call in tool_calls_list:
        function_name = tool_call["function"]["name"]
        tool_schema = next((t for t in tools if t["function"]["name"] == function_name), None)
        results_and_errors.append(
            use_tool_sync(tool_call["function"], tool_call["id"], tool_schema, function_dict=function_dict)
        )

    tool_results: list[MessageDict] = []
    tool_errors: list[MessageDict] = []

    for res, err in results_and_errors:
        tool_results.extend(res)
        tool_errors.extend(err)

    return new_history + tool_results + tool_errors


def use_tool_sync(
    tool_call: dict[str, Any],
    tool_id: str,
    tool_schema: dict[str, Any] | None,
    function_dict: FunctionDict | None = None,
) -> tuple[list[MessageDict], list[MessageDict]]:
    """Execute a single tool call (synchronous version).

    Args:
        tool_call: Tool call specification with name and arguments.
        tool_id: Unique identifier for this tool call.
        tool_schema: JSON schema for the tool (for error messages).
        function_dict: Mapping of function names to callables.

    Returns:
        Tuple of (results, errors) message lists.
    """
    if function_dict is None:
        function_dict = {}

    function_name = tool_call["name"]
    results: list[MessageDict] = []
    errors: list[MessageDict] = []

    if function_name not in function_dict:
        errors.append({"content": "ERROR", "role": "tool", "name": function_name, "tool_call_id": tool_id})
        errors.append({"content": "Only use a valid function in your function list.", "role": "system"})
        return results, errors

    called_function = function_dict[function_name]

    try:
        parsed_arguments = json.loads(tool_call["arguments"])
        try:
            # Call function synchronously
            result = called_function(**parsed_arguments)
            results.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": str(result),
                    "tool_call_id": tool_id,
                }
            )
        except Exception as e:
            error_str = (
                f"Error calling {function_name} function with passed arguments "
                f"{parsed_arguments}: {traceback.format_exc()} \n {e}"
            )
            errors.append(
                {
                    "content": error_str,
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": function_name,
                }
            )

    except json.JSONDecodeError:
        required_arguments = tool_schema["function"]["parameters"]["required"] if tool_schema else "unknown"
        if tool_call["arguments"] == "":
            error_content = (
                f"Your function call did not include any arguments. "
                f"Please try again with the correct arguments: {required_arguments}"
            )
        else:
            error_content = "Your function call did not parse as valid JSON. Please try again"
        errors.append({"content": "ERROR", "role": "tool", "name": function_name, "tool_call_id": tool_id})
        errors.append({"content": error_content, "role": "system"})

    return results, errors
