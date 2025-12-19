"""LLM client utilities for multi-provider API interactions.

This module provides the LLM client initialization and tool calling utilities
for both synchronous and asynchronous operations. It now supports multiple
providers (OpenAI, Anthropic, Google) through a unified interface.

Configuration:
    Set SCIENCEAI_LLM_PROVIDER environment variable to 'openai', 'anthropic', or 'google'
    Or create ~/Documents/ScienceAI/scienceai-config.json with provider settings
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import tiktoken

from .llm_providers import (
    MODEL_DEFAULT,
    MODEL_FAST,
    MODEL_REASONING,
    MODEL_VISION,
    ChatResponse,
    LLMConfig,
    LLMProvider,
    Provider,
    get_config,
    get_model_for_role,
    get_provider,
)

if TYPE_CHECKING:
    from threading import Event

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases for clarity
MessageDict = dict[str, Any]
ToolCallDict = dict[str, Any]
FunctionDict = dict[str, Callable[..., Any]]

# Global stop event for graceful shutdown
STOP_EVENT: Event | None = None


class LLMClientWrapper:
    """Wrapper that provides OpenAI-compatible interface for any provider.

    This maintains backward compatibility with existing code that uses
    client.chat.completions.create() syntax.
    """

    def __init__(self, is_async: bool = False):
        self._is_async = is_async
        self.chat = self._ChatNamespace(is_async)

    class _ChatNamespace:
        """Namespace for chat-related operations."""

        def __init__(self, is_async: bool):
            self._is_async = is_async
            self.completions = self._CompletionsNamespace(is_async)

        class _CompletionsNamespace:
            """Namespace for completions operations."""

            def __init__(self, is_async: bool):
                self._is_async = is_async

            def create(self, **kwargs: Any) -> Any:
                """Create a chat completion (sync) with interruptible polling."""
                # Check before starting
                if STOP_EVENT and STOP_EVENT.is_set():
                    raise SystemExit("Operation suppressed by STOP_EVENT")

                async def run_interruptible():
                    provider = get_provider()
                    # Use async implementation under the hood
                    task = asyncio.create_task(provider.chat_completion_async(**kwargs))

                    while not task.done():
                        if STOP_EVENT and STOP_EVENT.is_set():
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task
                            raise SystemExit("Operation suppressed by STOP_EVENT")

                        try:
                            # Wait for task or timeout
                            await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
                            return await task
                        except TimeoutError:
                            continue

                    return await task

                # Prepare coroutine object
                coro = run_interruptible()
                try:
                    # Run the async wrapper
                    response = asyncio.run(coro)
                    return _wrap_response(response)
                except RuntimeError as e:
                    # Fallback if we are already in an event loop
                    if "asyncio.run() cannot be called from a running event loop" in str(e):
                        # Close the unused coroutine to prevent RuntimeWarning
                        coro.close()

                        provider = get_provider()
                        response = provider.chat_completion(**kwargs)
                        return _wrap_response(response)
                    raise e
                except SystemExit:
                    raise
                except Exception as e:
                    raise e

            async def acreate(self, **kwargs: Any) -> Any:
                """Create a chat completion (async)."""
                if STOP_EVENT and STOP_EVENT.is_set():
                    raise SystemExit("Operation suppressed by STOP_EVENT")

                provider = get_provider()
                # Create task to allow polling
                task = asyncio.create_task(provider.chat_completion_async(**kwargs))

                while not task.done():
                    if STOP_EVENT and STOP_EVENT.is_set():
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                        raise SystemExit("Operation suppressed by STOP_EVENT")

                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
                        return _wrap_response(await task)
                    except TimeoutError:
                        continue

                return _wrap_response(await task)


class AsyncLLMClientWrapper(LLMClientWrapper):
    """Async version of the client wrapper."""

    def __init__(self):
        super().__init__(is_async=True)
        self.chat = self._AsyncChatNamespace()

    class _AsyncChatNamespace:
        """Async namespace for chat operations."""

        def __init__(self):
            self.completions = self._AsyncCompletionsNamespace()

        class _AsyncCompletionsNamespace:
            """Async namespace for completions."""

            def __init__(self):
                pass

            async def create(self, **kwargs: Any) -> Any:
                """Create a chat completion (async)."""
                if STOP_EVENT and STOP_EVENT.is_set():
                    raise SystemExit("Operation suppressed by STOP_EVENT")

                provider = get_provider()
                # Create task to allow polling
                task = asyncio.create_task(provider.chat_completion_async(**kwargs))

                while not task.done():
                    if STOP_EVENT and STOP_EVENT.is_set():
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                        raise SystemExit("Operation suppressed by STOP_EVENT")

                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
                        return _wrap_response(await task)
                    except TimeoutError:
                        continue

                return _wrap_response(await task)


class _WrappedResponse:
    """Wraps ChatResponse to provide OpenAI-compatible interface."""

    def __init__(self, response: ChatResponse):
        self._response = response
        self.choices = [self._Choice(response)]

    class _Choice:
        """Wrapper for response choice."""

        def __init__(self, response: ChatResponse):
            self.message = self._Message(response)
            self.finish_reason = response.finish_reason

        class _Message:
            """Wrapper for response message."""

            def __init__(self, response: ChatResponse):
                self.content = response.content
                self.tool_calls = None
                if response.tool_calls:
                    self.tool_calls = [_ToolCallWrapper(tc) for tc in response.tool_calls]
                self.thinking = response.thinking

            def __getitem__(self, key):
                """Support dict-like access."""
                mapping = {
                    "content": self.content,
                    "tool_calls": self.tool_calls,
                    "thinking": self.thinking,
                }
                if key in mapping:
                    return mapping[key]
                raise KeyError(key)

            def get(self, key, default=None):
                """Support dict-like get."""
                try:
                    return self[key]
                except KeyError:
                    return default


class _ToolCallWrapper:
    """Wrapper for tool calls to provide OpenAI-compatible interface."""

    def __init__(self, tool_call: dict[str, Any]):
        self.id = tool_call.get("id", "")
        self.type = tool_call.get("type", "function")
        self.function = self._Function(tool_call.get("function", {}))
        # Preserve thought_signature if present (required for Gemini 3 Pro with thinking)
        self.thought_signature = tool_call.get("thought_signature")

    class _Function:
        """Wrapper for function details."""

        def __init__(self, func: dict[str, Any]):
            self.name = func.get("name", "")
            self.arguments = func.get("arguments", "{}")


def _wrap_response(response: ChatResponse) -> _WrappedResponse:
    """Wrap a ChatResponse in OpenAI-compatible format."""
    return _WrappedResponse(response)


# Initialize provider and create wrapped clients
# Note: provider is now fetched dynamically by the client

# Async client for ingestion pipeline
async_client = AsyncLLMClientWrapper()

# Sync client for agents and data extraction
client = LLMClientWrapper()

# Token encoder for context management
# Note: This uses tiktoken which is OpenAI-specific, but token counting
# is approximate anyway and works well enough for context management
try:
    enc = tiktoken.encoding_for_model("gpt-4")
except Exception:
    # Fallback to cl100k_base encoding if model not found
    enc = tiktoken.get_encoding("cl100k_base")


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
    chat_response: _WrappedResponse | dict[str, Any],
    arguments: dict[str, Any],
    function_dict: FunctionDict | None = None,
    call_functions: bool = True,
    pre_tool_call: bool = False,
) -> list[MessageDict]:
    """Process and execute tool calls from a chat response (async version).

    Args:
        chat_response: Chat completion response or dict with tool_calls.
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
        thinking = chat_response.get("thinking")
    else:
        tool_calls = chat_response.choices[0].message.tool_calls
        content = chat_response.choices[0].message.content
        thinking = getattr(chat_response.choices[0].message, "thinking", None)

    tools = arguments.get("tools", [])
    tool_calls_list: list[ToolCallDict] = []

    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tc_entry = {
                    "function": {
                        "strict": True,
                        "arguments": tool_call["function"]["arguments"],
                        "name": tool_call["function"]["name"],
                    },
                    "id": tool_call["id"],
                    "type": "function",
                }
                # Preserve thought_signature if present
                if tool_call.get("thought_signature"):
                    tc_entry["thought_signature"] = tool_call["thought_signature"]
                tool_calls_list.append(tc_entry)
            else:
                tc_entry = {
                    "function": {
                        "arguments": tool_call.function.arguments,
                        "name": tool_call.function.name,
                    },
                    "id": tool_call.id,
                    "type": "function",
                }
                # Preserve thought_signature if present (from _ToolCallWrapper)
                if hasattr(tool_call, "thought_signature") and tool_call.thought_signature:
                    tc_entry["thought_signature"] = tool_call.thought_signature
                tool_calls_list.append(tc_entry)

    # Build assistant message with or without tool calls
    if call_functions:
        assistant_msg: MessageDict = {"content": content, "role": "assistant"}
        if tool_calls_list:
            assistant_msg["tool_calls"] = tool_calls_list
        if thinking:
            assistant_msg["thinking"] = thinking
        new_history: list[MessageDict] = [assistant_msg]
    else:
        # Should we handle calling functions = False case?
        # That branch constructs valid_calls which is just a list of calls, not messages.
        # But wait, lines 284-289 construct new_history regardless of whether we call functions later?
        # Ah, "if call_functions:" block at 284 handles history creation.
        pass  # Captured by replacement above

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
    if STOP_EVENT and STOP_EVENT.is_set():
        raise SystemExit("Operation suppressed by STOP_EVENT")

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
    chat_response: _WrappedResponse | dict[str, Any],
    arguments: dict[str, Any],
    function_dict: FunctionDict | None = None,
    call_functions: bool = True,
    pre_tool_call: bool = False,
) -> list[MessageDict]:
    """Process and execute tool calls from a chat response (synchronous version).

    Args:
        chat_response: Chat completion response or dict with tool_calls.
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
        thinking = chat_response.get("thinking")
    else:
        tool_calls = chat_response.choices[0].message.tool_calls
        content = chat_response.choices[0].message.content
        thinking = getattr(chat_response.choices[0].message, "thinking", None)

    tools = arguments.get("tools", [])
    tool_calls_list: list[ToolCallDict] = []

    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tc_entry = {
                    "function": {
                        "strict": True,
                        "arguments": tool_call["function"]["arguments"],
                        "name": tool_call["function"]["name"],
                    },
                    "id": tool_call["id"],
                    "type": "function",
                }
                # Preserve thought_signature if present
                if tool_call.get("thought_signature"):
                    tc_entry["thought_signature"] = tool_call["thought_signature"]
                tool_calls_list.append(tc_entry)
            else:
                tc_entry = {
                    "function": {
                        "arguments": tool_call.function.arguments,
                        "name": tool_call.function.name,
                    },
                    "id": tool_call.id,
                    "type": "function",
                }
                # Preserve thought_signature if present (from _ToolCallWrapper)
                if hasattr(tool_call, "thought_signature") and tool_call.thought_signature:
                    tc_entry["thought_signature"] = tool_call.thought_signature
                tool_calls_list.append(tc_entry)

    if call_functions:
        assistant_msg: MessageDict = {"content": content, "role": "assistant"}
        if tool_calls_list:
            assistant_msg["tool_calls"] = tool_calls_list
        if thinking:
            assistant_msg["thinking"] = thinking
            logger.info(
                f"use_tools_sync: Including thinking in assistant message (signature: {thinking.get('signature', 'N/A')[:30]}...)"
            )
        else:
            logger.debug("use_tools_sync: No thinking block in response")
        new_history: list[MessageDict] = [assistant_msg]

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
    if STOP_EVENT and STOP_EVENT.is_set():
        raise SystemExit("Operation suppressed by STOP_EVENT")

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


# Re-export provider utilities for convenience
__all__ = [
    "MODEL_DEFAULT",
    "MODEL_FAST",
    "MODEL_REASONING",
    "MODEL_VISION",
    "LLMConfig",
    "LLMProvider",
    "Provider",
    "async_client",
    "client",
    "enc",
    "get_config",
    "get_model_for_role",
    "get_provider",
    "trim_history",
    "update_stop_event",
    "use_tool",
    "use_tool_sync",
    "use_tools",
    "use_tools_sync",
]
