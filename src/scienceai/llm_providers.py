"""LLM Provider Abstraction Layer.

This module provides a unified interface for interacting with multiple LLM providers:
- OpenAI (GPT-4, GPT-4o, O1, etc.)
- Anthropic (Claude 3, Claude 3.5)
- Google (Gemini 1.5, Gemini 2.0)

Usage:
    from scienceai.llm_providers import get_provider, LLMConfig

    # Get the configured provider
    provider = get_provider()

    # Make a chat completion
    response = provider.chat_completion(messages, model=..., tools=...)
"""

from __future__ import annotations

# mypy: disable-error-code="unused-ignore"
import asyncio
import contextlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# Connection error retry constants
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds


def is_retryable_error(exception: Exception) -> bool:
    """Check if an exception is a retryable connection error."""
    # Check for common connection error types
    retryable_types = (
        ConnectionError,
        TimeoutError,
    )
    if isinstance(exception, retryable_types):
        return True

    # Check for httpx connection errors (used by Anthropic SDK)
    if type(exception).__name__ in ("ConnectError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError"):
        return True

    # Check for requests connection errors (used for REST calls)
    if type(exception).__module__.startswith("requests.exceptions"):
        return True

    # Check for API overload/rate limit errors that should be retried
    exception_str = str(exception).lower()
    if "overloaded" in exception_str or "rate limit" in exception_str or "529" in exception_str:
        return True

    return False


async def retry_async_call(coro_func, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """
    Execute an async coroutine function with retry logic for connection errors.

    Args:
        coro_func: Async function to call (NOT a coroutine - the function itself)
        *args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the successful call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if not is_retryable_error(e):
                raise

            if attempt < max_retries:
                wait_time = RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {type(e).__name__}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

    raise last_exception  # type: ignore


def retry_sync_call(func, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """
    Execute a sync function with retry logic for connection errors.

    Args:
        func: Sync function to call
        *args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the successful call

    Raises:
        The last exception if all retries fail
    """
    import time

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if not is_retryable_error(e):
                raise

            if attempt < max_retries:
                wait_time = RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {type(e).__name__}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

    raise last_exception  # type: ignore


class Provider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    ANTHROPIC_VERTEX = "anthropic-vertex"  # Claude on Google Vertex AI
    GOOGLE = "google"


@dataclass
class LLMConfig:
    """Configuration for LLM provider.

    Attributes:
        provider: The LLM provider to use (openai, anthropic, google)
        api_key: API key for the provider (if not set, will look in env vars)
        default_model: Default model to use for chat completions
        default_reasoning_model: Default model for reasoning/thinking tasks
        default_vision_model: Default model for vision tasks
        default_fast_model: Default model for fast/cheap tasks
    """

    provider: Provider = Provider.OPENAI
    api_key: str | None = None

    # GCP-specific configuration (for Google and Anthropic Vertex providers)
    gcp_service_account_path: str | None = None
    gcp_project_id: str | None = None
    gcp_region: str | None = None

    # Model mappings - these are the defaults, can be overridden per provider
    default_model: str = ""
    default_reasoning_model: str = ""
    default_vision_model: str = ""
    default_fast_model: str = ""

    # Context window limit for the default model (in tokens)
    context_limit: int = 0

    # Provider-specific model mappings
    model_aliases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Set default models based on provider."""
        if not self.default_model:
            self.default_model = self._get_default_model()
        if not self.default_reasoning_model:
            self.default_reasoning_model = self._get_default_reasoning_model()
        if not self.default_vision_model:
            self.default_vision_model = self._get_default_vision_model()
        if not self.default_fast_model:
            self.default_fast_model = self._get_default_fast_model()
        if not self.context_limit:
            self.context_limit = self._get_default_context_limit()

    def _get_default_model(self) -> str:
        """Get default model for the provider.

        As of December 2025:
        - OpenAI: gpt-5.2 (latest flagship model, released Nov 2025)
        - Anthropic: Claude Sonnet 4.5 (latest recommended model)
        - Google: Gemini 3 Pro (released Nov 18, 2025)
        """
        defaults = {
            Provider.OPENAI: "gpt-5.2",
            Provider.ANTHROPIC: "claude-sonnet-4-5",
            Provider.ANTHROPIC_VERTEX: "claude-sonnet-4-5",  # Same Claude models on Vertex
            Provider.GOOGLE: "gemini-3-pro-preview",
        }
        return defaults.get(self.provider, "gpt-5.2")

    def _get_default_reasoning_model(self) -> str:
        """Get default reasoning model for the provider.

        As of December 2025:
        - OpenAI: o4-mini (multimodal reasoning, released Apr 2025)
        - Anthropic: Claude Opus 4.5 (most capable reasoning)
        - Google: Gemini 3 Pro (with thinking mode)
        """
        defaults = {
            Provider.OPENAI: "o4-mini",
            Provider.ANTHROPIC: "claude-opus-4-5",  # Uses extended thinking
            Provider.ANTHROPIC_VERTEX: "claude-opus-4-5",  # Uses extended thinking
            Provider.GOOGLE: "gemini-3-pro-preview",  # Supports thinking mode
        }
        return defaults.get(self.provider, "o4-mini")

    def _get_default_vision_model(self) -> str:
        """Get default vision model for the provider.

        As of December 2025:
        - OpenAI: gpt-4o (omni model with best vision capabilities)
        - Anthropic: Claude Sonnet 4.5 (excellent vision support)
        - Google: Gemini 3 Pro (multimodal with 1M context)
        """
        defaults = {
            Provider.OPENAI: "gpt-4o",
            Provider.ANTHROPIC: "claude-sonnet-4-5",
            Provider.ANTHROPIC_VERTEX: "claude-sonnet-4-5",
            Provider.GOOGLE: "gemini-3-pro-preview",
        }
        return defaults.get(self.provider, "gpt-4o")

    def _get_default_fast_model(self) -> str:
        """Get default fast/cheap model for the provider.

        As of December 2025:
        - OpenAI: gpt-5-mini (fast version of GPT-5)
        - Anthropic: Claude Haiku 4.5 (fastest Claude model)
        - Google: Gemini 2.5 Flash-Lite (fastest/cheapest Gemini)
        """
        defaults = {
            Provider.OPENAI: "gpt-5-mini",
            Provider.ANTHROPIC: "claude-haiku-4-5",
            Provider.ANTHROPIC_VERTEX: "claude-haiku-4-5",
            Provider.GOOGLE: "gemini-3-pro-preview",
        }
        return defaults.get(self.provider, "gpt-5-mini")

    def _get_default_context_limit(self) -> int:
        """Get default context window limit for the provider's default model.

        As of December 2025:
        - OpenAI gpt-5.2: 400,000 tokens
        - Anthropic Claude Sonnet 4.5: 200,000 tokens (1M with beta header)
        - Google Gemini 3 Pro: 1,000,000 tokens
        """
        defaults = {
            Provider.OPENAI: 400_000,  # GPT-5.2 context limit
            Provider.ANTHROPIC: 200_000,  # Claude Sonnet 4.5 default context
            Provider.ANTHROPIC_VERTEX: 200_000,  # Same as direct Anthropic API
            Provider.GOOGLE: 1_000_000,  # Gemini 3 Pro context limit
        }
        return defaults.get(self.provider, 400_000)


@dataclass
class ChatMessage:
    """Unified chat message format."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]] | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    thinking: dict[str, str] | None = None

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI message format."""
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic message format."""
        msg: dict[str, Any] = {"role": self.role}

        # Anthropic doesn't have a system role in messages - it's a separate param
        if self.role == "system":
            # Will be handled separately
            return {}

        # Handle tool role -> Anthropic uses 'user' with tool_result content blocks
        if self.role == "tool":
            msg["role"] = "user"
            msg["content"] = [{"type": "tool_result", "tool_use_id": self.tool_call_id, "content": self.content or ""}]
            return msg

        # Handle assistant with tool calls -> Anthropic uses content blocks
        if self.tool_calls:
            thinking_blocks = []
            other_blocks = []

            # Add thinking from separate field first (if present)
            if self.thinking and self.thinking.get("signature"):
                thinking_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": self.thinking.get("thinking", ""),
                        "signature": self.thinking.get("signature"),
                    }
                )
                logger.debug(f"to_anthropic_format: Added thinking block (has {len(self.tool_calls)} tool_calls)")
            else:
                logger.debug(f"to_anthropic_format: NO thinking for assistant w/tool_calls. thinking={self.thinking}")

            # Handle content - it can be a string OR a list of blocks
            if self.content:
                if isinstance(self.content, list):
                    # Content is a list of blocks - need to separate thinking from other blocks
                    for item in self.content:
                        if isinstance(item, dict):
                            if item.get("type") in ("thinking", "redacted_thinking"):
                                thinking_blocks.append(item)
                            elif item.get("type") == "text":
                                other_blocks.append({"type": "text", "text": item.get("text", "")})
                            else:
                                # Pass through other block types (but not tool_use - we add those separately)
                                if item.get("type") != "tool_use":
                                    other_blocks.append(item)
                        elif isinstance(item, str):
                            other_blocks.append({"type": "text", "text": item})
                else:
                    # Content is a simple string
                    other_blocks.append({"type": "text", "text": self.content})

            # Add tool calls
            for tc in self.tool_calls:
                other_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    }
                )

            # CRITICAL: Thinking blocks MUST come first per Anthropic API requirements
            msg["content"] = thinking_blocks + other_blocks
            return msg

        # Handle image content or already-list content (e.g., from DB storage)
        if isinstance(self.content, list):
            thinking_blocks = []
            other_blocks = []

            # If we have thinking in the separate field, add it first
            if self.thinking and self.thinking.get("signature"):
                thinking_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": self.thinking.get("thinking", ""),
                        "signature": self.thinking.get("signature"),
                    }
                )

            # Process items, separating thinking from other blocks
            for item in self.content:
                if item.get("type") == "text":
                    other_blocks.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("data:"):
                        # Parse data URL
                        media_type = url.split(";")[0].split(":")[1]
                        base64_data = url.split(",")[1]
                        other_blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": media_type, "data": base64_data},  # type: ignore
                            }
                        )
                    else:
                        # URL-based image - Anthropic requires base64
                        other_blocks.append({"type": "text", "text": f"[Image URL: {url}]"})
                elif item.get("type") in ("thinking", "redacted_thinking"):
                    # Collect thinking blocks - they MUST come first
                    thinking_blocks.append(item)
                elif item.get("type") == "tool_use":
                    # Tool use blocks must come after thinking and text
                    other_blocks.append(item)
                else:
                    # Pass through any other block types
                    other_blocks.append(item)

            # CRITICAL: Thinking blocks MUST be first per Anthropic API requirements
            msg["content"] = thinking_blocks + other_blocks
        else:
            if self.thinking and self.thinking.get("signature"):
                content_blocks = [
                    {
                        "type": "thinking",
                        "thinking": self.thinking.get("thinking", ""),
                        "signature": self.thinking.get("signature"),
                    }
                ]
                if self.content:
                    content_blocks.append({"type": "text", "text": self.content or ""})
                msg["content"] = content_blocks
            else:
                msg["content"] = self.content or ""

        return msg

    def to_google_format(self) -> dict[str, Any]:
        """Convert to Google Gemini message format."""
        # Gemini uses 'model' instead of 'assistant'
        role = "model" if self.role == "assistant" else self.role

        # Gemini doesn't have a system role in content - it's a separate param
        if self.role == "system":
            return {}

        parts = []

        # Handle tool role -> Gemini uses functionResponse
        if self.role == "tool":
            parts.append(
                {
                    "functionResponse": {
                        "name": self.name or "unknown",
                        "response": {"result": self.content or ""},
                    }
                }
            )
            return {"role": role, "parts": parts}

        # Handle tool calls -> Gemini uses functionCall
        if self.tool_calls:
            if self.content:
                parts.append({"text": self.content})  # type: ignore
            for i, tc in enumerate(self.tool_calls):
                function_call_part = {
                    "functionCall": {
                        "name": tc.get("function", {}).get("name", ""),
                        "args": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    }
                }
                # Include thought_signature if present (required for Gemini 3 Pro with thinking)
                # The signature is only on the first function call in parallel calls
                # Signatures are stored as base64 strings and need to be decoded back to bytes
                import base64

                sig_value = tc.get("thought_signature")
                if not sig_value and i == 0 and self.thinking:
                    sig_value = self.thinking.get("signature")

                if sig_value and sig_value != "skip_thought_signature_validator":
                    # Decode base64 string back to bytes for API
                    try:
                        function_call_part["thought_signature"] = base64.b64decode(sig_value)  # type: ignore[assignment]
                    except Exception:
                        # If decoding fails, use as-is (might already be the skip value)
                        function_call_part["thought_signature"] = sig_value  # type: ignore[assignment]
                elif sig_value == "skip_thought_signature_validator":
                    function_call_part["thought_signature"] = sig_value  # type: ignore[assignment]
                else:
                    # For historical messages without signatures, use the validator skip
                    # This is a last resort per Gemini docs but necessary for backward compatibility
                    function_call_part["thought_signature"] = "skip_thought_signature_validator"  # type: ignore[assignment]
                    logger.warning(
                        f"No thought_signature for function call '{tc.get('function', {}).get('name', 'unknown')}' - "
                        "using skip_thought_signature_validator fallback (may impact model performance)"
                    )
                parts.append(function_call_part)
            return {"role": role, "parts": parts}

        # Handle image content
        if isinstance(self.content, list):
            for item in self.content:
                if item.get("type") == "text":
                    parts.append({"text": item["text"]})
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("data:"):
                        media_type = url.split(";")[0].split(":")[1]
                        base64_data = url.split(",")[1]
                        parts.append({"inlineData": {"mimeType": media_type, "data": base64_data}})
                    else:
                        parts.append({"text": f"[Image URL: {url}]"})  # type: ignore
        else:
            parts.append({"text": self.content or ""})  # type: ignore

        return {"role": role, "parts": parts}


@dataclass
class ChatResponse:
    """Unified chat response format."""

    content: str | None
    tool_calls: list[dict[str, Any]] | None
    finish_reason: str
    raw_response: Any  # Original provider response for debugging
    thinking: dict[str, str] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)


@dataclass
class ToolSchema:
    """Unified tool schema format (OpenAI-compatible)."""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = False

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_google_format(self) -> dict[str, Any]:
        """Convert to Google Gemini tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Make a synchronous chat completion request."""
        pass

    @abstractmethod
    async def chat_completion_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Make an asynchronous chat completion request."""
        pass

    def resolve_model(self, model: str | None) -> str:
        """Resolve model name using aliases and defaults."""
        if model is None:
            return self.config.default_model

        # Check for aliases
        if model in self.config.model_aliases:
            return self.config.model_aliases[model]

        return model

    def _normalize_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Normalize tool format - subclasses can override."""
        return tools

    @abstractmethod
    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import AsyncOpenAI, OpenAI

        api_key = config.api_key or self._get_api_key()
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment or config file."""
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
                    return openai_key  # type: ignore
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read OpenAI API key from config file: {e}")

        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
            "or add it to ~/Documents/ScienceAI/scienceai-keys.json"
        )

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Make a synchronous chat completion request to OpenAI."""
        resolved_model = self.resolve_model(model)

        request_args: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
        }
        extra_body: dict[str, Any] = {}
        if tools:
            request_args["tools"] = tools
        if tool_choice:
            request_args["tool_choice"] = tool_choice
        if temperature is not None:
            request_args["temperature"] = temperature
        if max_tokens is not None:
            extra_body["max_completion_tokens"] = max_tokens
        if reasoning_effort is not None:
            extra_body["reasoning_effort"] = reasoning_effort
        if tools and not parallel_tool_calls:
            request_args["parallel_tool_calls"] = False
        if extra_body:
            request_args["extra_body"] = extra_body

        # Add any additional kwargs
        request_args.update(kwargs)

        response = self.client.chat.completions.create(**request_args)

        # Convert to unified format
        message = response.choices[0].message
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]

        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop",
            raw_response=response,
        )

    async def chat_completion_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Make an asynchronous chat completion request to OpenAI."""
        resolved_model = self.resolve_model(model)

        request_args: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
        }

        extra_body: dict[str, Any] = {}
        if tools:
            request_args["tools"] = tools
        if tool_choice:
            request_args["tool_choice"] = tool_choice
        if temperature is not None:
            request_args["temperature"] = temperature
        if max_tokens is not None:
            extra_body["max_completion_tokens"] = max_tokens
        if reasoning_effort is not None:
            extra_body["reasoning_effort"] = reasoning_effort
        if tools and not parallel_tool_calls:
            request_args["parallel_tool_calls"] = False
        if extra_body:
            request_args["extra_body"] = extra_body

        request_args.update(kwargs)

        response = await self.async_client.chat.completions.create(**request_args)

        message = response.choices[0].message
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]

        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop",
            raw_response=response,
        )

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens using tiktoken (OpenAI)."""
        import tiktoken

        model = self.config.default_model or "gpt-4"
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if (key == "content" and value) or (key == "name" and value):
                    num_tokens += len(encoding.encode(str(value)))
                elif key == "tool_calls" and value:
                    num_tokens += len(encoding.encode(json.dumps(value)))

        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Try GCP service account first (Vertex AI)
        gcp_config = load_gcp_config("anthropic_vertex")
        if gcp_config and os.path.exists(gcp_config.get("service_account_path", "")):
            # Use Vertex AI with service account
            logger.info("Using Claude via Vertex AI (service account)")
            self._init_vertex_client(gcp_config)
        else:
            # Fall back to direct Anthropic API
            logger.info("Using Claude via Anthropic API key")
            import anthropic

            api_key = config.api_key or self._get_api_key()
            self.client = anthropic.Anthropic(api_key=api_key)
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key)

    def _init_vertex_client(self, gcp_config: dict):
        """Initialize Vertex AI client for Claude using AnthropicVertex."""
        from anthropic import AnthropicVertex, AsyncAnthropicVertex

        sa_path = gcp_config["service_account_path"]
        project_id = gcp_config["project_id"]
        # Force global region as requested
        region = "global"

        # Set credential file path for Google Auth
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

        self.client = AnthropicVertex(project_id=project_id, region=region)  # type: ignore
        self.async_client = AsyncAnthropicVertex(project_id=project_id, region=region)  # type: ignore

        # Store for token counting (REST API)
        self._project_id = project_id
        self._region = region
        self._sa_path = sa_path

    def _get_api_key(self) -> str:
        """Get Anthropic API key from environment or config file."""
        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            return api_key

        base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
        target_key = os.path.join(base_key_path, "scienceai-keys.json")

        if os.path.exists(target_key):
            try:
                with open(target_key) as file:
                    key_list = json.load(file)
                if anthropic_key := key_list.get("anthropic"):
                    return anthropic_key  # type: ignore
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read Anthropic API key from config file: {e}")

        raise ValueError(
            "Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable "
            "or add it to ~/Documents/ScienceAI/scienceai-keys.json"
        )

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]], set[int]]:
        """Convert messages to Anthropic format, extracting system message.

        Returns:
            Tuple of (system_message, converted_messages, compressed_indices)
            where compressed_indices are the indices in converted_messages that are compressed.
        """
        system_message = None
        converted: list[dict[str, Any]] = []
        compressed_indices: set[int] = set()

        for msg in messages:
            chat_msg = ChatMessage(
                role=msg.get("role", "user"),
                content=msg.get("content"),
                name=msg.get("name"),
                tool_call_id=msg.get("tool_call_id"),
                tool_calls=msg.get("tool_calls"),
                thinking=msg.get("thinking"),
            )

            if chat_msg.role == "system":
                system_message = chat_msg.content if isinstance(chat_msg.content, str) else str(chat_msg.content)
            else:
                converted_msg = chat_msg.to_anthropic_format()
                if converted_msg:  # Skip empty messages
                    # Track if this is a compressed message
                    if msg.get("compressed"):
                        compressed_indices.add(len(converted))
                    converted.append(converted_msg)

        return system_message, converted, compressed_indices

    def _strip_thinking_except_last(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
        """Strip thinking blocks from all messages except the last one that has thinking.

        This is used for context optimization - when context is above 50%, we strip
        thinking from older messages to reduce token count while keeping the most recent.

        Args:
            messages: List of messages in Anthropic format (already converted)

        Returns:
            Tuple of (stripped_messages, last_thinking_msg_idx, stripped_count)
            - stripped_messages: Deep copy with thinking stripped from all but last
            - last_thinking_msg_idx: Index of the message that kept thinking (-1 if none)
            - stripped_count: Number of messages that had thinking stripped
        """
        import copy

        result = copy.deepcopy(messages)

        # Find the last message with thinking
        last_thinking_msg_idx = -1
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"):
                        last_thinking_msg_idx = i
                        break
                if last_thinking_msg_idx != -1:
                    break

        # Strip thinking from all messages except the last one that has it
        stripped_count = 0
        for i, msg in enumerate(result):
            if i == last_thinking_msg_idx:
                continue
            content = msg.get("content")
            if isinstance(content, list):
                original_len = len(content)
                msg["content"] = [
                    block
                    for block in content
                    if not (isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"))
                ]
                if len(msg["content"]) < original_len:
                    stripped_count += 1

        return result, last_thinking_msg_idx, stripped_count

    def _prepare_messages_for_thinking(
        self,
        converted_messages: list[dict[str, Any]],
        context_limit: int = 0,
        system_message: str | None = None,
    ) -> list[dict[str, Any]]:
        """Prepare converted messages for thinking-enabled API calls.

        This method handles all transformations required for thinking mode:
        1. Bootstrap: Convert assistant messages without thinking-first to user role
        2. Context optimization: Strip thinking from older messages when above 50% context
        3. Final safety check: Ensure no assistant messages have thinking in wrong position

        Args:
            converted_messages: Messages already converted to Anthropic format
            context_limit: Context window limit for optimization (0 = skip optimization)
            system_message: System prompt (for token counting during optimization)

        Returns:
            Transformed messages ready for thinking-enabled API call
        """
        import copy

        messages = copy.deepcopy(converted_messages)  # Don't mutate original

        # STEP 1: BOOTSTRAP - Convert trailing assistant messages without thinking to user role
        # We scan backwards from the end and convert assistant messages until we find one with thinking.
        # This "bootstraps" thinking behavior by ensuring the last assistant message(s) have thinking.
        # If the last assistant message already has thinking, n=0 and nothing is converted.
        converted_count = 0
        stripped_tool_use_ids: set[str] = set()

        # Find indices of assistant messages to convert (from end backwards until one has thinking)
        indices_to_convert: list[int] = []
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                has_thinking_first = False
                if isinstance(content, list) and len(content) > 0:
                    first_block = content[0]
                    if isinstance(first_block, dict) and first_block.get("type") in ("thinking", "redacted_thinking"):
                        has_thinking_first = True

                if has_thinking_first:
                    # Found an assistant message with thinking - stop here
                    break
                else:
                    # This assistant message doesn't have thinking - mark for conversion
                    indices_to_convert.append(i)

        # Convert the marked messages
        for i in indices_to_convert:
            msg = messages[i]
            content = msg.get("content", "")
            messages[i]["role"] = "user"
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        stripped_tool_use_ids.add(block.get("id", ""))
                messages[i]["content"] = [
                    block for block in content if not (isinstance(block, dict) and block.get("type") == "tool_use")
                ]
            converted_count += 1

        # Strip orphaned tool_result blocks
        if stripped_tool_use_ids:
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    msg["content"] = [
                        block
                        for block in content
                        if not (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and block.get("tool_use_id") in stripped_tool_use_ids
                        )
                    ]

        # Merge consecutive user messages
        if converted_count > 0:
            merged_messages: list[dict[str, Any]] = []
            for msg in messages:
                if merged_messages and merged_messages[-1].get("role") == "user" and msg.get("role") == "user":
                    prev_content = merged_messages[-1].get("content", "")
                    curr_content = msg.get("content", "")
                    if isinstance(prev_content, str) and isinstance(curr_content, str):
                        merged_messages[-1]["content"] = prev_content + "\n\n---\n\n" + curr_content
                    elif isinstance(prev_content, list) and isinstance(curr_content, list):
                        merged_messages[-1]["content"] = prev_content + curr_content
                    elif isinstance(prev_content, str) and isinstance(curr_content, list):
                        merged_messages[-1]["content"] = [{"type": "text", "text": prev_content}, *curr_content]
                    elif isinstance(prev_content, list) and isinstance(curr_content, str):
                        merged_messages[-1]["content"] = [*prev_content, {"type": "text", "text": curr_content}]
                else:
                    merged_messages.append(msg)
            messages = merged_messages
            logger.info(f"THINKING BOOTSTRAP: Converted {converted_count} assistant message(s) to 'user' role.")

        # STEP 2: CONTEXT OPTIMIZATION - Strip thinking from older messages when above 50%
        if messages and context_limit > 0:
            try:
                messages_for_counting = []
                if system_message:
                    messages_for_counting.append({"role": "system", "content": system_message})
                messages_for_counting.extend(messages)

                token_count = self.count_tokens(messages_for_counting, raw=True)
                context_usage = token_count / context_limit

                if context_usage > 0.5:
                    # Use shared helper to strip thinking
                    messages, last_thinking_msg_idx, stripped_count = self._strip_thinking_except_last(messages)

                    if stripped_count > 0:
                        logger.info(
                            f"CONTEXT OPTIMIZATION: Stripped thinking from {stripped_count} messages "
                            f"(context at {context_usage:.1%}, kept idx {last_thinking_msg_idx})."
                        )

                        # After stripping, assistant messages that lost thinking need conversion to user
                        # (because they may now have text/tool_use first instead of thinking)
                        post_converted = 0
                        post_stripped_ids: set[str] = set()
                        for i in range(len(messages)):
                            if i == last_thinking_msg_idx:
                                continue
                            msg = messages[i]
                            if msg.get("role") == "assistant":
                                content = msg.get("content")
                                if isinstance(content, list):
                                    # Check if first block is not thinking (was stripped or never had it)
                                    first_is_thinking = (
                                        len(content) > 0
                                        and isinstance(content[0], dict)
                                        and content[0].get("type") in ("thinking", "redacted_thinking")
                                    )
                                    if not first_is_thinking:
                                        messages[i]["role"] = "user"
                                        for block in content:
                                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                                post_stripped_ids.add(block.get("id", ""))
                                        messages[i]["content"] = [
                                            block
                                            for block in content
                                            if not (isinstance(block, dict) and block.get("type") == "tool_use")
                                        ]
                                        post_converted += 1

                        # Strip orphaned tool_results
                        if post_stripped_ids:
                            for msg in messages:
                                content = msg.get("content")
                                if isinstance(content, list):
                                    msg["content"] = [
                                        block
                                        for block in content
                                        if not (
                                            isinstance(block, dict)
                                            and block.get("type") == "tool_result"
                                            and block.get("tool_use_id") in post_stripped_ids
                                        )
                                    ]

                        # Merge consecutive user messages
                        if post_converted > 0:
                            merged: list[dict[str, Any]] = []
                            for msg in messages:
                                if merged and merged[-1].get("role") == "user" and msg.get("role") == "user":
                                    prev = merged[-1].get("content", "")
                                    curr = msg.get("content", "")
                                    if isinstance(prev, list) and isinstance(curr, list):
                                        merged[-1]["content"] = prev + curr
                                    elif isinstance(prev, str) and isinstance(curr, str):
                                        merged[-1]["content"] = prev + "\n\n---\n\n" + curr
                                    elif isinstance(prev, str) and isinstance(curr, list):
                                        merged[-1]["content"] = [{"type": "text", "text": prev}, *curr]
                                    elif isinstance(prev, list) and isinstance(curr, str):
                                        merged[-1]["content"] = [*prev, {"type": "text", "text": curr}]
                                else:
                                    merged.append(msg)
                            messages = merged
                            logger.info(f"CONTEXT OPT POST-FIX: Converted {post_converted} more assistant message(s).")

            except Exception as e:
                logger.warning(f"Context optimization failed: {e}")

        # STEP 3: FINAL SAFETY CHECK - Ensure no problematic assistant messages remain
        problem_msgs = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, list) and len(content) > 0:
                    first_type = content[0].get("type") if isinstance(content[0], dict) else None
                    has_thinking = any(
                        isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in content
                    )
                    if has_thinking and first_type not in ("thinking", "redacted_thinking"):
                        problem_msgs.append(f"idx={i}, first={first_type}")

        if problem_msgs:
            logger.warning(
                f"THINKING SAFETY: Found {len(problem_msgs)} problematic messages after prep: {problem_msgs[:5]}"
            )

        return messages

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert tools to Anthropic format."""
        if not tools:
            return None

        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
        return converted

    def _convert_tool_choice(self, tool_choice: dict[str, Any] | str | None) -> dict[str, Any] | None:
        """Convert tool_choice to Anthropic format."""
        if tool_choice is None:
            return None
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return {"type": "none"}  # Anthropic doesn't support "none", but we can try
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                return {"type": "tool", "name": tool_choice["function"]["name"]}
        return {"type": "auto"}

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Make a synchronous chat completion request."""
        # Use common API implementation for both Direct and Vertex clients
        return self._chat_completion_api(
            messages,
            model,
            tools,
            tool_choice,
            temperature,
            max_tokens,
            reasoning_effort,
            parallel_tool_calls,
            **kwargs,
        )

    def _chat_completion_api(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat completion using Anthropic API (Direct or Vertex)."""
        resolved_model = self.resolve_model(model)
        system_message, converted_messages, _compressed_indices = self._convert_messages(messages)

        request_args: dict[str, Any] = {
            "model": resolved_model,
            "messages": converted_messages,
            "max_tokens": max_tokens or 20000,
        }

        if system_message:
            request_args["system"] = system_message
        if tools:
            request_args["tools"] = self._convert_tools(tools)
        if tool_choice:
            converted_choice = self._convert_tool_choice(tool_choice)
            if converted_choice:
                # Apply strictly single tool use constraint if requested
                if parallel_tool_calls is False and converted_choice.get("type") == "auto":
                    converted_choice["disable_parallel_tool_use"] = True
                request_args["tool_choice"] = converted_choice
        elif parallel_tool_calls is False and tools:
            # If no explicit choice but parallel is disabled, default to auto with parallel disabled
            request_args["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if temperature is not None:
            request_args["temperature"] = temperature

        # Handle extended thinking for reasoning models
        if reasoning_effort:
            # Claude uses extended thinking via budget_tokens
            effort_map = {"low": 2048, "medium": 8192, "high": 16384}
            requested_budget = effort_map.get(reasoning_effort, 2048)

            # Ensure budget fits within max_tokens with a buffer for the response
            # Default buffer for response content
            response_buffer = 2048
            current_max_tokens = request_args["max_tokens"]

            if requested_budget >= current_max_tokens:
                # Cap the budget to leave room for response
                # This handles the Vertex AI 8192 limit vs 16384 high budget case
                # 8192 - 2048 = 6144, which is a "decent amount"
                adjusted_budget = max(1024, current_max_tokens - response_buffer)

                logger.info(
                    f"Adjusting reasoning budget from {requested_budget} to {adjusted_budget} "
                    f"to fit within max_tokens {current_max_tokens}"
                )
                request_args["thinking"] = {"type": "enabled", "budget_tokens": adjusted_budget}
            else:
                request_args["thinking"] = {"type": "enabled", "budget_tokens": requested_budget}

            # Use shared helper for all thinking-related transformations
            converted_messages = self._prepare_messages_for_thinking(
                converted_messages,
                context_limit=self.config.context_limit,
                system_message=system_message,
            )
            # Update request_args with the transformed messages
            request_args["messages"] = converted_messages

            # SAFETY CHECK: Anthropic does not support 'thinking' with forced tool use.
            # If tool_choice is explicit (anything other than 'auto'), we explicitly disable thinking.
            # Convert explicitly checks for 'auto' type, so if it's not 'auto' and present, it's forced.
            tool_choice_arg = request_args.get("tool_choice")
            if "thinking" in request_args and tool_choice_arg:
                tc_type = tool_choice_arg.get("type")
                if tc_type != "auto":
                    logger.warning(
                        f"Disabling thinking because tool_choice forces tool use (type='{tc_type}'). "
                        "Thinking is only supported with tool_choice='auto'."
                    )
                    request_args.pop("thinking", None)
                elif tc_type == "auto" and tool_choice_arg.get("disable_parallel_tool_use"):
                    # Note: tool_choice='auto' IS supported with thinking, even if parallel is disabled.
                    pass

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
                    # If content becomes empty (e.g. only had thinking), we might have an issue.
                    # Anthropic doesn't like empty content. But if it had thinking, it usually has tool_use or text too.
                    # If it was ONLY thinking, and we strip it, we have empty content.
                    # We should probably replace with a dummy text or something, but usually stripping thinking is fine if tool calls exist.
                    msg["content"] = new_content

        # DEBUG: Log whether thinking is enabled
        if "thinking" in request_args:
            logger.info(f"[SYNC API] Making request with thinking ENABLED: {request_args['thinking']}")

            # DIAGNOSTIC: Scan for problematic assistant messages
            problem_msgs = []
            for i, msg in enumerate(converted_messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if isinstance(content, list) and len(content) > 0:
                        first_type = (
                            content[0].get("type") if isinstance(content[0], dict) else type(content[0]).__name__
                        )
                        has_thinking = any(
                            isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in content
                        )
                        if has_thinking and first_type not in ("thinking", "redacted_thinking"):
                            problem_msgs.append(f"idx={i}, first={first_type}")
            if problem_msgs:
                logger.warning(
                    f"[SYNC API] DIAGNOSTIC: Found {len(problem_msgs)} problematic assistant messages: {problem_msgs[:5]}"
                )
            else:
                logger.info("[SYNC API] DIAGNOSTIC: All assistant messages pass thinking-first check")
            # Enable interleaved thinking for proper thinking between tool calls
            # This beta header ensures Claude produces thinking blocks consistently during tool-use loops
            # Must use client.beta.messages.create() to pass beta headers
            request_args["betas"] = ["interleaved-thinking-2025-05-14"]
            response = retry_sync_call(self.client.beta.messages.create, **request_args)
        else:
            logger.info("[SYNC API] Making request with thinking DISABLED")
            response = retry_sync_call(self.client.messages.create, **request_args)

        # Convert response to unified format
        content = None
        tool_calls = None
        thinking = None

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "thinking":
                thinking = {"thinking": block.thinking, "signature": block.signature}
            elif block.type == "redacted_thinking":
                thinking = {"thinking": "[Redacted by Anthropic]", "signature": block.data}
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": json.dumps(block.input)},
                    }
                )

        if thinking:
            logger.info(
                f"[SYNC API] Captured thinking block with signature: {thinking.get('signature', 'N/A')[:50]}..."
            )
        else:
            content_types = [block.type for block in response.content]
            logger.warning(f"[SYNC API] NO thinking block in response! Content types: {content_types}")

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "end_turn",
            raw_response=response,
            thinking=thinking,
        )

    def count_tokens(self, messages: list[dict[str, Any]], raw: bool = False) -> int:
        """Count tokens using Anthropic API (Beta for Direct, REST for Vertex).

        Args:
            messages: List of message dicts to count tokens for
            raw: If True, return raw token count. If False (default), return count
                 assuming thinking blocks will be stripped when >50% context usage
                 (which matches what chat_completion will actually send).
        """
        # CONTEXT OPTIMIZATION: When raw=False, check if we're above 50% context
        # If so, return the count assuming thinking blocks will be stripped (matching chat_completion behavior)
        if not raw and self.config.context_limit > 0:
            # First get the raw count using recursive call
            raw_count = self.count_tokens(messages, raw=True)
            context_usage = raw_count / self.config.context_limit

            if context_usage > 0.5:
                # Use shared helper to strip thinking
                optimized_messages, _last_thinking_msg_idx, stripped_count = self._strip_thinking_except_last(messages)

                # Return the optimized count
                optimized_count = self.count_tokens(optimized_messages, raw=True)
                logger.debug(
                    f"count_tokens: Context at {context_usage:.1%}, returning optimized count "
                    f"({optimized_count} vs raw {raw_count}, stripped {stripped_count})"
                )
                return optimized_count

        resolved_model = self.resolve_model(None)

        # Check if using Vertex client (indicated by presence of _project_id set in _init_vertex_client)
        if getattr(self, "_project_id", None):
            # Vertex AI Token Counting (REST API)
            try:
                import requests
                from google.auth.transport.requests import Request
                from google.oauth2 import service_account

                # Authenticate
                # Try to use stored SA path first, otherwise rely on default env
                if hasattr(self, "_sa_path") and self._sa_path:
                    creds = service_account.Credentials.from_service_account_file(
                        self._sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                else:
                    # Fallback to default credentials
                    from google.auth import default

                    creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

                creds.refresh(Request())
                access_token = creds.token

                # Determine region for token counting
                # Global inference endpoint doesn't support token counting, try us-east5 or us-central1
                count_region = getattr(self, "_region", "us-east5")
                if count_region == "global" or count_region == "us-east4":  # Known failing regions
                    count_region = "us-east5"

                host = f"{count_region}-aiplatform.googleapis.com"
                project_id = getattr(self, "_project_id", None)

                if not project_id:
                    logger.warning("Project ID not found for Vertex token counting.")
                    raise ValueError("Project ID missing")

                base_url = f"https://{host}/v1/projects/{project_id}/locations/{count_region}/publishers/anthropic/models/count-tokens:rawPredict"

                headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json; charset=utf-8"}

                # Convert messages for the body
                # Vertex rawPredict expects { "model": ..., "messages": ... }
                # We need to minimally process messages to ensure they match expected format
                # The _convert_messages method tailored for SDK might need slight adjustment or raw usage
                # But for simple counting, let's use the converted messages from _convert_messages (list of dicts)
                _, converted_messages, _ = self._convert_messages(messages)

                # SANITIZATION: Vertex countTokens is strict about tool_use -> tool_result pairing.
                # If a tool_use is dangling (interrupted or followed by text), it fails with 400.
                # We iterate and inject dummy results for any missing ones.
                sanitized_messages = []
                i = 0
                while i < len(converted_messages):
                    msg = converted_messages[i]
                    sanitized_messages.append(msg)

                    # Check if this message has tool_use
                    tool_use_ids = []
                    if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                        for block in msg["content"]:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                tool_use_ids.append(block["id"])

                    if tool_use_ids:
                        # Look ahead for tool_result
                        next_msg = converted_messages[i + 1] if i + 1 < len(converted_messages) else None

                        # Identify missing IDs
                        missing_ids = set(tool_use_ids)

                        if next_msg and next_msg.get("role") == "user":
                            # Check existing results in next message
                            if isinstance(next_msg.get("content"), list):
                                for block in next_msg["content"]:
                                    if isinstance(block, dict) and block.get("type") == "tool_result":
                                        missing_ids.discard(block.get("tool_use_id"))

                        # If we have missing IDs, we need to fix
                        if missing_ids:
                            dummy_results = [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tid,
                                    "content": "Dummy result for token counting",
                                }
                                for tid in missing_ids
                            ]

                            if next_msg and next_msg.get("role") == "user":
                                # Append to next message
                                # Ensure content is list
                                content = next_msg.get("content")
                                if isinstance(content, str):
                                    next_msg["content"] = [{"type": "text", "text": content}]
                                elif content is None:
                                    next_msg["content"] = []

                                # Prepend dummy results to ensure they come first
                                # (Vertex might require tool_result to be the first block if mixed with text)
                                next_msg["content"] = dummy_results + next_msg["content"]
                                # Next message will be processed in next iteration (already in converted_messages)
                            else:
                                # Insert new message
                                new_msg = {"role": "user", "content": dummy_results}
                                # We need to insert into the stream we are processing?
                                # Actually we are appending to sanitized_messages.
                                # If next_msg existed but wasn't user (e.g. assistant), we insert BEFORE it.
                                # If next_msg didn't exist, we just append.
                                sanitized_messages.append(new_msg)

                    i += 1

                # Use shared helper for thinking message transformations (mirrors chat_completion exactly)
                # Note: We don't pass context_limit here to avoid recursive token counting
                sanitized_messages = self._prepare_messages_for_thinking(
                    sanitized_messages,
                    context_limit=0,  # Skip context optimization in token counting to avoid recursion
                    system_message=None,
                )

                data = {
                    "model": "claude-3-5-sonnet-v2@20241022",
                    "messages": sanitized_messages,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                }

                response = requests.post(base_url, headers=headers, json=data, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    return result.get("input_tokens", 0)  # type: ignore
                else:
                    logger.warning(f"Vertex token counting failed with {response.status_code}: {response.text}")
                    # Fallback to estimate
            except Exception as e:
                logger.warning(f"Vertex token counting error: {e}")

        else:
            # Direct Anthropic API Token Counting
            try:
                system_message, converted_messages, _ = self._convert_messages(messages)
                request_args = {
                    "model": resolved_model,
                    "messages": converted_messages,
                }
                if system_message:
                    request_args["system"] = system_message

                # logger.info(f"Anthropic token counting request: {str(self.client.beta.messages.count_tokens)}")
                response = self.client.beta.messages.count_tokens(**request_args)  # type: ignore
                return response.input_tokens  # type: ignore
            except Exception as e:
                logger.warning(f"Anthropic beta token counting failed: {e}")

        # Return None when token counting fails - caller should skip context emission
        return None  # type: ignore

    async def chat_completion_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Make an asynchronous chat completion request to Anthropic."""
        resolved_model = self.resolve_model(model)
        system_message, converted_messages, _compressed_indices = self._convert_messages(messages)

        request_args: dict[str, Any] = {
            "model": resolved_model,
            "messages": converted_messages,
            "max_tokens": max_tokens or 20000,
        }

        if system_message:
            request_args["system"] = system_message
        if tools:
            request_args["tools"] = self._convert_tools(tools)
        if tool_choice:
            converted_choice = self._convert_tool_choice(tool_choice)
            if converted_choice:
                # Apply strictly single tool use constraint if requested
                if parallel_tool_calls is False and converted_choice.get("type") == "auto":
                    converted_choice["disable_parallel_tool_use"] = True
                request_args["tool_choice"] = converted_choice
        elif parallel_tool_calls is False and tools:
            # If no explicit choice but parallel is disabled, default to auto with parallel disabled
            request_args["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}

        if temperature is not None:
            request_args["temperature"] = temperature

        if reasoning_effort:
            effort_map = {"low": 1024, "medium": 4096, "high": 16384}
            requested_budget = effort_map.get(reasoning_effort, 4096)
            current_max_tokens = request_args.get("max_tokens", 8192)

            # Ensure budget < max_tokens
            if requested_budget >= current_max_tokens:
                # Cap the budget to leave room for response
                # This handles the Vertex AI 8192 limit vs 16384 high budget case
                # 8192 - 2048 = 6144, which is a "decent amount"
                response_buffer = 2048
                adjusted_budget = max(1024, current_max_tokens - response_buffer)

                logger.info(
                    f"Adjusting reasoning budget from {requested_budget} to {adjusted_budget} "
                    f"to fit within max_tokens {current_max_tokens}"
                )
                request_args["thinking"] = {"type": "enabled", "budget_tokens": adjusted_budget}
            else:
                request_args["thinking"] = {"type": "enabled", "budget_tokens": requested_budget}

            # Use shared helper for all thinking-related transformations
            converted_messages = self._prepare_messages_for_thinking(
                converted_messages,
                context_limit=self.config.context_limit,
                system_message=system_message,
            )
            # Update request_args with the transformed messages
            request_args["messages"] = converted_messages

            # SAFETY CHECK: Anthropic does not support 'thinking' with forced tool use.
            # If tool_choice is explicit (anything other than 'auto'), we explicitly disable thinking.
            tool_choice_arg = request_args.get("tool_choice")
            if "thinking" in request_args and tool_choice_arg:
                tc_type = tool_choice_arg.get("type")
                if tc_type != "auto":
                    logger.warning(
                        f"Disabling thinking because tool_choice forces tool use (type='{tc_type}'). "
                        "Thinking is only supported with tool_choice='auto'."
                    )
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

        # Enable interleaved thinking for proper thinking between tool calls
        # Must use client.beta.messages.create() to pass beta headers
        if "thinking" in request_args:
            request_args["betas"] = ["interleaved-thinking-2025-05-14"]
            response = await retry_async_call(self.async_client.beta.messages.create, **request_args)
        else:
            response = await retry_async_call(self.async_client.messages.create, **request_args)

        content: str | None = None
        tool_calls: list[dict[str, Any]] | None = None
        thinking = None

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "thinking":
                thinking = {"thinking": block.thinking, "signature": block.signature}
            elif block.type == "redacted_thinking":
                thinking = {"thinking": "[Redacted by Anthropic]", "signature": block.data}
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": json.dumps(block.input)},
                    }
                )

        if thinking:
            logger.info(f"Captured thinking block (async) with signature: {thinking.get('signature', 'N/A')[:50]}...")

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "end_turn",
            raw_response=response,
            thinking=thinking,
        )


class GoogleProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Try GCP service account first (priority for production/enterprise)
        gcp_config = load_gcp_config("google_gcp")
        if gcp_config and os.path.exists(gcp_config.get("service_account_path", "")):
            # Use Vertex AI with service account
            logger.info("Using Google Gemini via Vertex AI (service account)")
            self._use_vertex = True
            self._init_vertex_client(gcp_config)
        else:
            # Fall back to regular API key
            logger.info("Using Google Gemini via API key")
            self._use_vertex = False
            from google import genai

            api_key = config.api_key or self._get_api_key()
            self.client = genai.Client(api_key=api_key)

    def _init_vertex_client(self, gcp_config: dict):
        """Initialize Vertex AI client with service account using google.genai SDK.

        This uses the unified google.genai SDK with vertexai=True, which provides
        full ThinkingConfig support for Vertex AI (same as the API key path).
        """
        from google import genai
        from google.auth import load_credentials_from_file
        from google.genai.types import HttpOptions

        sa_path = gcp_config["service_account_path"]
        project_id = gcp_config["project_id"]
        # Force global region as used by Vertex AI for Gemini
        region = "global"

        # Load credentials from service account file
        credentials, _ = load_credentials_from_file(sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.credentials = credentials
        self.project_id = project_id
        self.region = region

        # Create google.genai Client with Vertex AI enabled
        # This uses the same SDK as the API key path but with service account auth
        self.client = genai.Client(
            credentials=credentials,
            vertexai=True,
            project=project_id,
            location=region,
            http_options=HttpOptions(api_version="v1"),
        )
        logger.info(f"Initialized google.genai client for Vertex AI (project: {project_id}, region: {region})")

    def _get_api_key(self) -> str:
        """Get Google API key from environment or config file."""
        if api_key := os.environ.get("GOOGLE_API_KEY"):
            return api_key
        if api_key := os.environ.get("GEMINI_API_KEY"):
            return api_key

        base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
        target_key = os.path.join(base_key_path, "scienceai-keys.json")

        if os.path.exists(target_key):
            try:
                with open(target_key) as file:
                    key_list = json.load(file)
                if google_key := key_list.get("google"):
                    return google_key  # type: ignore
                if google_key := key_list.get("gemini"):
                    return google_key  # type: ignore
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read Google API key from config file: {e}")

        raise ValueError(
            "Google API key not found. Please set the GOOGLE_API_KEY environment variable "
            "or add it to ~/Documents/ScienceAI/scienceai-keys.json"
        )

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages to Google Gemini format (for API), extracting system instruction."""
        system_instruction = None
        converted = []

        for msg in messages:
            chat_msg = ChatMessage(
                role=msg.get("role", "user"),
                content=msg.get("content"),
                name=msg.get("name"),
                tool_call_id=msg.get("tool_call_id"),
                tool_calls=msg.get("tool_calls"),
                thinking=msg.get("thinking"),  # Pass thinking for thought_signature access
            )

            if chat_msg.role == "system":
                system_instruction = chat_msg.content if isinstance(chat_msg.content, str) else str(chat_msg.content)
            else:
                converted_msg = chat_msg.to_google_format()
                if converted_msg:
                    converted.append(converted_msg)

        return system_instruction, converted

    def _convert_messages_vertex(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[Any]]:
        """Convert messages to Vertex AI format using proper Part and Content objects."""
        from vertexai.generative_models import Content, Part

        system_instruction = None
        converted = []

        for msg in messages:
            role = msg.get("role", "user")

            # Extract system message
            if role == "system":
                content = msg.get("content")
                system_instruction = content if isinstance(content, str) else str(content)
                continue

            # Convert role: "assistant" -> "model", "tool" -> "user"
            vertex_role = "model" if role == "assistant" else "user"

            parts = []

            # Handle tool response
            if role == "tool":
                # For tool responses, just convert to text since the SDK doesn't have direct support
                tool_result = msg.get("content", "")
                tool_name = msg.get("name", "tool")
                parts.append(Part.from_text(f"[Tool {tool_name} returned: {tool_result}]"))
            # Handle tool calls from assistant
            elif msg.get("tool_calls"):
                # Add text content if present
                if msg.get("content"):
                    parts.append(Part.from_text(str(msg["content"])))

                # Add function calls as text representation
                for tc in msg["tool_calls"]:
                    func_name = tc.get("function", {}).get("name", "")
                    func_args = tc.get("function", {}).get("arguments", "{}")
                    parts.append(Part.from_text(f"[Calling function {func_name} with args: {func_args}]"))
            # Handle regular content (text or multimodal)
            elif isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        parts.append(Part.from_text(item["text"]))
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            import base64

                            media_type = url.split(";")[0].split(":")[1]
                            base64_data = url.split(",")[1]
                            image_bytes = base64.b64decode(base64_data)
                            parts.append(Part.from_data(data=image_bytes, mime_type=media_type))
                        else:
                            # For URL-based images, add a text placeholder
                            parts.append(Part.from_text(f"[Image URL: {url}]"))
            else:
                # Simple text content
                content_text = msg.get("content", "")
                if content_text:
                    parts.append(Part.from_text(str(content_text)))

            if parts:
                converted.append(Content(role=vertex_role, parts=parts))

        return system_instruction, converted

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert tools to Google Gemini format."""
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                function_declarations.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {"type": "object", "properties": {}}),
                    }
                )

        return [{"function_declarations": function_declarations}]

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Make a synchronous chat completion request to Google Gemini."""
        # Auto-redirect Claude models to AnthropicProvider
        # This handles configuration mismatches where Google provider is active but Claude model is requested
        resolved_model = self.resolve_model(model)
        if "claude" in resolved_model.lower():
            logger.info(
                f"Detected Claude model '{resolved_model}' in GoogleProvider. Redirecting to AnthropicProvider."
            )
            # Instantiate Anthropic provider with same config
            # Note: It will load its own specific GCP config if needed
            anthropic_provider = AnthropicProvider(self.config)
            return anthropic_provider.chat_completion(
                messages,
                model,
                tools,
                tool_choice,
                temperature,
                max_tokens,
                reasoning_effort,
                parallel_tool_calls,
                **kwargs,
            )

        # Both Vertex and API key paths now use the same google.genai Client
        # so we use the unified _chat_completion_api method for both
        return self._chat_completion_api(
            messages,
            model,
            tools,
            tool_choice,
            temperature,
            max_tokens,
            reasoning_effort,
            parallel_tool_calls,
            **kwargs,
        )

    def _chat_completion_api(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat completion using regular Google AI API."""
        from google.genai import types

        resolved_model = self.resolve_model(model)
        system_instruction, converted_messages = self._convert_messages(messages)

        # Build generation config
        generation_config = types.GenerateContentConfig()
        if temperature is not None:
            generation_config.temperature = temperature
        if max_tokens is not None:
            generation_config.max_output_tokens = max_tokens

        # Convert and pass tools to the API
        converted_tools = self._convert_tools(tools) if tools else None
        if converted_tools:
            generation_config.tools = converted_tools  # type: ignore
            logger.debug(
                f"Google AI API: Passing {len(converted_tools[0]['function_declarations'])} tools to model {resolved_model}"
            )

            # Handle tool_choice to control function calling behavior
            # - tool_choice={"type": "function", "function": {"name": "X"}} -> ANY + allowed_function_names
            # - tool_choice="required" or "any" -> ANY (force function call)
            # - tool_choice="none" -> NONE
            # - tool_choice="auto" or None -> AUTO
            if tool_choice:
                calling_mode = "AUTO"
                allowed_functions = None

                if isinstance(tool_choice, dict):
                    if tool_choice.get("type") == "function":
                        # Force a specific function to be called
                        func_name = tool_choice.get("function", {}).get("name")
                        if func_name:
                            calling_mode = "ANY"
                            allowed_functions = [func_name]
                            logger.info(f"Google AI API: tool_choice forces function '{func_name}' - using ANY mode")
                elif isinstance(tool_choice, str):
                    if tool_choice.lower() in ("required", "any"):
                        calling_mode = "ANY"
                        logger.info(f"Google AI API: tool_choice='{tool_choice}' - using ANY mode")
                    elif tool_choice.lower() == "none":
                        calling_mode = "NONE"
                        logger.info("Google AI API: tool_choice='none' - using NONE mode")

                # Build ToolConfig using proper types from google.genai
                if calling_mode != "AUTO":
                    fc_config = types.FunctionCallingConfig(mode=calling_mode)
                    if allowed_functions:
                        fc_config = types.FunctionCallingConfig(
                            mode=calling_mode, allowed_function_names=allowed_functions
                        )
                    generation_config.tool_config = types.ToolConfig(function_calling_config=fc_config)
                    logger.debug(f"  ToolConfig mode: {calling_mode}, allowed: {allowed_functions}")
        else:
            logger.debug(f"Google AI API: No tools passed to model {resolved_model}")

        # Handle system instruction
        if system_instruction:
            generation_config.system_instruction = system_instruction

        # Handle reasoning/thinking mode
        if reasoning_effort:
            # Gemini 2.0+ Flash Thinking uses special thinking config
            effort_map = {"low": 1024, "medium": 8192, "high": 24576}
            budget = effort_map.get(reasoning_effort, 8192)
            # Enable include_thoughts to get the thinking content in the response
            generation_config.thinking_config = types.ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=True,
            )

        # Disable automatic function calling - we handle function execution manually
        # This prevents the SDK's AFC loop from interfering with our control flow
        generation_config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)

        # Build and execute the request with retry for empty responses
        max_empty_retries = 1
        response = None

        for attempt in range(max_empty_retries + 1):
            response = self.client.models.generate_content(
                model=resolved_model,
                contents=converted_messages,  # type: ignore
                config=generation_config,
            )

            # Check if response has usable content
            has_content = (
                response.candidates and response.candidates[0].content and response.candidates[0].content.parts
            )

            if has_content:
                break  # Got valid response

            if attempt < max_empty_retries:
                # Log and retry
                block_reason = None
                finish_reason_raw = None
                if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                    block_reason = getattr(response.prompt_feedback, "block_reason", None)
                if response.candidates and response.candidates[0]:
                    finish_reason_raw = getattr(response.candidates[0], "finish_reason", None)
                logger.warning(
                    f"Google AI: Empty response (finish_reason={finish_reason_raw}, block_reason={block_reason}), "
                    f"retrying ({attempt + 1}/{max_empty_retries})..."
                )
                import time

                time.sleep(0.5)  # Brief backoff before retry

        # Convert response to unified format
        content: str | None = None
        tool_calls: list[dict[str, Any]] | None = None
        thinking_content: str | None = None
        thought_signature: str | None = None

        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:  # type: ignore
                # Check for thought parts first (Gemini marks thinking with 'thought' boolean)
                if hasattr(part, "thought") and part.thought:
                    # This is a thinking/thought part
                    if hasattr(part, "text") and part.text:
                        thinking_content = part.text
                        logger.debug(f"Google AI: Extracted thinking content ({len(part.text)} chars)")
                elif hasattr(part, "text") and part.text:
                    content = part.text
                    # Check for thought_signature on text parts (for non-function call responses)
                    if hasattr(part, "thought_signature") and part.thought_signature:
                        # Signature is bytes, encode to base64 for JSON serialization
                        import base64

                        sig = part.thought_signature
                        thought_signature = base64.b64encode(sig).decode("utf-8") if isinstance(sig, bytes) else sig
                        logger.debug("Google AI: Extracted thought_signature from text part")
                elif hasattr(part, "function_call") and part.function_call:
                    if tool_calls is None:
                        tool_calls = []
                    fc = part.function_call
                    tool_call_entry = {
                        "id": f"call_{len(tool_calls)}",  # Gemini doesn't provide IDs
                        "type": "function",
                        "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args))},  # type: ignore
                    }
                    # Extract thought_signature from function call part if present
                    # Only the first function call in parallel calls will have the signature
                    if hasattr(part, "thought_signature") and part.thought_signature:
                        # Signature is bytes, encode to base64 for JSON serialization
                        import base64

                        sig = part.thought_signature
                        sig_encoded = base64.b64encode(sig).decode("utf-8") if isinstance(sig, bytes) else sig
                        tool_call_entry["thought_signature"] = sig_encoded
                        thought_signature = sig_encoded  # Also store in thinking dict
                        logger.debug("Google AI: Extracted thought_signature from function_call part")
                    tool_calls.append(tool_call_entry)

        # Log if we still have an empty response after retries
        if (
            not response
            or not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            block_reason = None
            if response and hasattr(response, "prompt_feedback") and response.prompt_feedback:
                block_reason = getattr(response.prompt_feedback, "block_reason", None)
            if response and response.candidates and response.candidates[0]:
                finish_reason_raw = getattr(response.candidates[0], "finish_reason", None)
                logger.warning(
                    f"Google AI: Response still empty after retry (finish_reason={finish_reason_raw}, block_reason={block_reason})"
                )
            else:
                logger.warning(f"Google AI: No candidates after retry (block_reason={block_reason})")

        finish_reason = "stop"
        if response and response.candidates:
            finish_reason = (
                str(response.candidates[0].finish_reason) if response.candidates[0].finish_reason else "stop"
            )

        # Enforce sequential tool calling: only return first tool call when parallel is disabled
        if not parallel_tool_calls and tool_calls and len(tool_calls) > 1:
            logger.info(f"Google AI: Limiting {len(tool_calls)} tool calls to 1 (parallel_tool_calls=False)")
            tool_calls = [tool_calls[0]]

        # Build thinking dict if we captured thinking content or signature
        thinking: dict[str, str] | None = None
        if thinking_content or thought_signature:
            thinking = {}
            if thinking_content:
                thinking["thinking"] = thinking_content
            if thought_signature:
                thinking["signature"] = thought_signature  # Store signature for replay

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_response=response,
            thinking=thinking,
        )

    def _chat_completion_vertex(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat completion using Vertex AI."""
        from vertexai.generative_models import GenerationConfig, Tool

        resolved_model = self.resolve_model(model)
        system_instruction, converted_messages = self._convert_messages_vertex(messages)

        # Build generation config for Vertex AI
        config_params = {}
        if temperature is not None:
            config_params["temperature"] = temperature
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens

        # Handle reasoning/thinking mode for Vertex AI
        # Note: Vertex SDK doesn't have direct ThinkingConfig, but thinking is enabled
        # automatically for supported models. We configure the model to return thoughts.
        if reasoning_effort:
            # For Vertex AI, thinking is primarily controlled at the model level
            # The thinking budget could be set if Vertex SDK supported it directly
            # For now, we just enable thinking extraction in the response
            logger.info(f"Vertex AI: Thinking mode enabled with reasoning_effort={reasoning_effort}")

        generation_config = GenerationConfig(**config_params) if config_params else None  # type: ignore

        # Convert tools to Vertex AI format if provided
        vertex_tools = None
        tool_config = None
        tool_count = 0
        if tools:
            from vertexai.generative_models import FunctionDeclaration, ToolConfig

            function_declarations = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    function_declarations.append(
                        FunctionDeclaration(
                            name=func["name"],
                            description=func.get("description", ""),
                            parameters=func.get("parameters", {"type": "object", "properties": {}}),
                        )
                    )
            if function_declarations:
                tool_count = len(function_declarations)
                vertex_tools = [Tool(function_declarations=function_declarations)]

                # Translate tool_choice to Gemini's function calling mode
                # - tool_choice={"type": "function", "function": {"name": "X"}} -> ANY + allowed_function_names
                # - tool_choice="required" or "any" -> ANY (force function call)
                # - tool_choice="none" -> NONE
                # - tool_choice="auto" or None -> AUTO
                calling_mode = ToolConfig.FunctionCallingConfig.Mode.AUTO
                allowed_functions = None

                if isinstance(tool_choice, dict):
                    if tool_choice.get("type") == "function":
                        # Force a specific function to be called
                        func_name = tool_choice.get("function", {}).get("name")
                        if func_name:
                            calling_mode = ToolConfig.FunctionCallingConfig.Mode.ANY
                            allowed_functions = [func_name]
                            logger.info(f"Vertex AI: tool_choice forces function '{func_name}' - using ANY mode")
                elif isinstance(tool_choice, str):
                    if tool_choice.lower() in ("required", "any"):
                        calling_mode = ToolConfig.FunctionCallingConfig.Mode.ANY
                        logger.info(f"Vertex AI: tool_choice='{tool_choice}' - using ANY mode")
                    elif tool_choice.lower() == "none":
                        calling_mode = ToolConfig.FunctionCallingConfig.Mode.NONE
                        logger.info("Vertex AI: tool_choice='none' - using NONE mode")

                # Build the ToolConfig with appropriate mode
                if allowed_functions:
                    tool_config = ToolConfig(
                        function_calling_config=ToolConfig.FunctionCallingConfig(
                            mode=calling_mode,
                            allowed_function_names=allowed_functions,
                        )
                    )
                    logger.debug(f"  ToolConfig mode: {calling_mode}, allowed_function_names: {allowed_functions}")
                else:
                    tool_config = ToolConfig(
                        function_calling_config=ToolConfig.FunctionCallingConfig(
                            mode=calling_mode,
                        )
                    )
                    logger.debug(f"  ToolConfig mode: {calling_mode}")

        # Debug log to verify tools are being passed
        if vertex_tools:
            logger.info(f"Vertex AI: Passing {tool_count} tools to model {resolved_model} (with ToolConfig)")
        else:
            logger.info(f"Vertex AI: No tools passed to model {resolved_model}")

        # Create model instance
        from vertexai.generative_models import GenerativeModel

        vertex_model = GenerativeModel(resolved_model, system_instruction=system_instruction)

        logger.info(f"Vertex AI: Making generate_content call with {len(converted_messages)} messages")

        # Make the request with tool_config to enforce function calling
        response = vertex_model.generate_content(
            contents=converted_messages,
            generation_config=generation_config,
            tools=vertex_tools,
            tool_config=tool_config,
        )

        logger.info("Vertex AI: Response received, processing...")

        # Debug: Log response structure
        if response.candidates:
            logger.debug(f"  Candidates count: {len(response.candidates)}")
            if response.candidates[0].content:
                logger.debug(f"  Parts count: {len(response.candidates[0].content.parts)}")
                for i, part in enumerate(response.candidates[0].content.parts):
                    part_attrs = [attr for attr in dir(part) if not attr.startswith("_")]
                    logger.debug(f"  Part {i}: available attributes: {part_attrs[:10]}")  # First 10 to avoid spam
                    if hasattr(part, "text"):
                        logger.debug(f"  Part {i}: HAS text attribute")
                    if hasattr(part, "function_call"):
                        logger.debug(f"  Part {i}: HAS function_call attribute")
                    if hasattr(part, "thought"):
                        logger.debug(f"  Part {i}: HAS thought attribute = {part.thought}")

        # Convert response (Vertex AI response format is similar)
        content: str | None = None
        tool_calls: list[dict[str, Any]] | None = None
        thinking_content: str | None = None

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                # Check for thought parts first (Vertex AI marks thinking with 'thought' boolean)
                if hasattr(part, "thought") and part.thought:
                    # This is a thinking/thought part
                    if hasattr(part, "text") and part.text:
                        thinking_content = part.text
                        logger.debug(f"Vertex AI: Extracted thinking content ({len(part.text)} chars)")
                elif hasattr(part, "text") and part.text:
                    content = part.text
                    logger.info(f"Vertex AI: Got TEXT content (length: {len(content or '')} chars)")
                elif hasattr(part, "function_call") and part.function_call:
                    if tool_calls is None:
                        tool_calls = []
                    fc = part.function_call
                    tool_calls.append(
                        {
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args))},
                        }
                    )
                    logger.info(f"Vertex AI: Got FUNCTION_CALL: {fc.name}")

        # SELF-CORRECTION: Detect text-based function calls and retry with correction message
        # Pattern: [Calling function FUNC_NAME with args: {...}]
        import re

        text_func_pattern = r"\[Calling function (\w+) with args: (\{.*?\})\]"

        if content and not tool_calls and re.search(text_func_pattern, content, re.DOTALL):
            logger.warning(
                "Vertex AI: Detected malformed text-based function call, attempting retry with correction..."
            )

            # Build correction message
            correction_message = (
                "Your previous response described a function call as text instead of actually calling it. "
                "You wrote something like '[Calling function X with args: {...}]' as text. "
                "This is incorrect. You must USE the function by making an actual function call, not describe it. "
                "Please try again and actually invoke the function using the proper function calling mechanism."
            )

            # Append the malformed assistant response and correction to messages for retry
            retry_messages = list(converted_messages)  # Copy
            from vertexai.generative_models import Content, Part

            # Add the malformed response as assistant
            retry_messages.append(Content(role="model", parts=[Part.from_text(content)]))
            # Add correction as user
            retry_messages.append(Content(role="user", parts=[Part.from_text(correction_message)]))

            # Retry the request
            try:
                logger.info("Vertex AI: Retrying with correction message...")
                retry_response = vertex_model.generate_content(
                    contents=retry_messages,
                    generation_config=generation_config,
                    tools=vertex_tools,
                    tool_config=tool_config,
                )

                # Check if retry succeeded with proper function call
                retry_content = None
                retry_tool_calls: list[dict[str, Any]] | None = None

                if retry_response.candidates and retry_response.candidates[0].content:
                    for part in retry_response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            retry_content = part.text
                        elif hasattr(part, "function_call") and part.function_call:
                            if retry_tool_calls is None:
                                retry_tool_calls = []
                            fc = part.function_call
                            retry_tool_calls.append(
                                {
                                    "id": f"call_{len(retry_tool_calls)}",
                                    "type": "function",
                                    "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args))},
                                }
                            )

                # If retry got a proper function call, use it (as if original worked)
                if retry_tool_calls:
                    logger.info(
                        f"Vertex AI: Retry succeeded! Got proper FUNCTION_CALL: {retry_tool_calls[0]['function']['name']}"
                    )
                    content = retry_content
                    tool_calls = retry_tool_calls
                else:
                    logger.warning("Vertex AI: Retry still returned text, falling back to parsing original response")
                    # Fall through to parsing fallback below

            except Exception as e:
                logger.warning(f"Vertex AI: Retry failed with error: {e}, falling back to parsing original response")

        # FINAL FALLBACK: If still no tool_calls, try to parse text-based function calls
        if content and not tool_calls:
            matches = re.findall(text_func_pattern, content, re.DOTALL)
            if matches:
                logger.info(f"Vertex AI: Parsing {len(matches)} text-based function call(s) as final fallback")
                tool_calls = []
                for func_name, args_str in matches:
                    try:
                        # Validate it's valid JSON
                        json.loads(args_str)
                        tool_calls.append(
                            {
                                "id": f"call_{len(tool_calls)}",
                                "type": "function",
                                "function": {"name": func_name, "arguments": args_str},
                            }
                        )
                        logger.info(f"Vertex AI: Parsed text-based function call: {func_name}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Vertex AI: Failed to parse args for {func_name}: {e}")

                # Remove the function call text from content, leaving just the preamble
                if tool_calls:
                    first_match_pos = content.find("[Calling function")
                    if first_match_pos > 0:
                        content = content[:first_match_pos].strip()
                    else:
                        content = None  # Only had function call text

        finish_reason = "stop"
        if response.candidates:
            finish_reason = (
                str(response.candidates[0].finish_reason) if response.candidates[0].finish_reason else "stop"
            )

        # Enforce sequential tool calling: only return first tool call when parallel is disabled
        if not parallel_tool_calls and tool_calls and len(tool_calls) > 1:
            logger.info(f"Vertex AI: Limiting {len(tool_calls)} tool calls to 1 (parallel_tool_calls=False)")
            tool_calls = [tool_calls[0]]

        # Build thinking dict if we captured thinking content
        thinking: dict[str, str] | None = None
        if thinking_content:
            thinking = {"thinking": thinking_content}

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_response=response,
            thinking=thinking,
        )

    async def chat_completion_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Make an asynchronous chat completion request to Google Gemini."""

        # Auto-redirect Claude models to AnthropicProvider
        resolved_model = self.resolve_model(model)
        if "claude" in resolved_model.lower():
            logger.info(
                f"Detected Claude model '{resolved_model}' in GoogleProvider. Redirecting to AnthropicProvider."
            )
            anthropic_provider = AnthropicProvider(self.config)
            return await anthropic_provider.chat_completion_async(
                messages,
                model,
                tools,
                tool_choice,
                temperature,
                max_tokens,
                reasoning_effort,
                parallel_tool_calls,
                **kwargs,
            )

        # Google's genai library doesn't have native async support
        # We run the sync version in an executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chat_completion(
                messages=messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                parallel_tool_calls=parallel_tool_calls,
                **kwargs,
            ),
        )

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens using Google/Gemini API."""
        resolved_model = self.resolve_model(None)

        try:
            # Both Vertex and API key paths now use the same google.genai Client (self.client)
            system_instruction, converted_messages = self._convert_messages(messages)
            response = self.client.models.count_tokens(
                model=resolved_model,
                contents=converted_messages,  # type: ignore
                config={"system_instruction": system_instruction} if system_instruction else None,
            )
            return response.total_tokens or 0  # type: ignore
        except Exception as e:
            logger.warning(f"Google token counting failed: {e}")
            char_count = sum(len(str(m.get("content", ""))) for m in messages)
            return char_count // 4


# Global configuration and provider instance
_config: LLMConfig | None = None
_provider: LLMProvider | None = None


def load_config() -> LLMConfig:
    """Load LLM configuration from environment variables and config file."""
    # Check environment variable first
    provider_str = os.environ.get("SCIENCEAI_LLM_PROVIDER", "openai").lower()

    try:
        provider = Provider(provider_str)
    except ValueError:
        logger.warning(f"Unknown provider '{provider_str}', defaulting to OpenAI")
        provider = Provider.OPENAI

    # Load from config file if exists
    # Priority 1: Current working directory (Local)
    # Priority 2: ~/Documents/ScienceAI (Global)

    config_paths = [
        os.path.join(os.getcwd(), "scienceai-config.json"),
        os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI", "scienceai-config.json"),
    ]

    config_data = {}

    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    # We only load the first one we find that is valid
                    config_data = json.load(f)
                    logger.info(f"Loaded config from: {config_path}")
                    break
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read config file at {config_path}: {e}")

    # Override provider from config file if present
    if "provider" in config_data:
        with contextlib.suppress(ValueError):
            provider = Provider(config_data["provider"].lower())

    logger.info(f"Loaded LLM Configuration: Provider={provider.value}")

    return LLMConfig(
        provider=provider,
        api_key=config_data.get("api_key"),
        default_model=config_data.get("default_model", ""),
        default_reasoning_model=config_data.get("default_reasoning_model", ""),
        default_vision_model=config_data.get("default_vision_model", ""),
        default_fast_model=config_data.get("default_fast_model", ""),
        model_aliases=config_data.get("model_aliases", {}),
    )


def save_config(config: LLMConfig) -> bool:
    """Save LLM configuration to config file.

    Args:
        config: Configuration to save

    Returns:
        True if successful, False otherwise
    """
    base_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
    os.makedirs(base_path, exist_ok=True)
    config_path = os.path.join(base_path, "scienceai-config.json")

    try:
        # Load existing config to preserve other settings
        config_data = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_data = json.load(f)

        # Update provider selection
        config_data["provider"] = config.provider.value

        # Save back
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved provider selection: {config.provider.value}")
        return True
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Failed to save config: {e}")
        return False


def get_config() -> LLMConfig:
    """Get the current LLM configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: LLMConfig) -> None:
    """Set the LLM configuration."""
    global _config, _provider
    _config = config
    _provider = None  # Reset provider to force re-initialization


def get_provider() -> LLMProvider:
    """Get the configured LLM provider instance."""
    global _provider
    if _provider is None:
        config = get_config()
        _provider = create_provider(config)
    return _provider


def create_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider instance from configuration."""
    if config.provider == Provider.OPENAI:
        return OpenAIProvider(config)
    elif config.provider == Provider.ANTHROPIC:
        return AnthropicProvider(config)
    elif config.provider == Provider.GOOGLE:
        return GoogleProvider(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def get_provider_type() -> Provider:
    """Get the current provider type."""
    return get_config().provider


# Model role constants for use throughout the codebase
# These map to the appropriate model for each use case
MODEL_DEFAULT = "default"
MODEL_REASONING = "reasoning"
MODEL_VISION = "vision"
MODEL_FAST = "fast"


def get_model_for_role(role: str) -> str:
    """Get the appropriate model name for a given role.

    Args:
        role: One of 'default', 'reasoning', 'vision', 'fast'

    Returns:
        The model name to use
    """
    config = get_config()
    role_map = {
        MODEL_DEFAULT: config.default_model,
        MODEL_REASONING: config.default_reasoning_model,
        MODEL_VISION: config.default_vision_model,
        MODEL_FAST: config.default_fast_model,
    }
    return role_map.get(role, config.default_model)


def get_context_limit() -> int:
    """Get the context window limit for the current provider's default model.

    Returns:
        Context limit in tokens
    """
    config = get_config()
    return config.context_limit


def get_available_providers() -> dict[str, bool]:
    """Check which providers have API keys configured.

    Returns:
        Dictionary mapping provider names to availability status.
    """
    available = {}

    # Check OpenAI
    openai_available = bool(os.environ.get("OPENAI_API_KEY"))
    if not openai_available:
        base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
        target_key = os.path.join(base_key_path, "scienceai-keys.json")
        if os.path.exists(target_key):
            try:
                with open(target_key) as file:
                    key_list = json.load(file)
                openai_available = bool(key_list.get("openai"))
            except (json.JSONDecodeError, OSError):
                pass
    available["openai"] = openai_available

    # Check Anthropic
    anthropic_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not anthropic_available:
        base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
        target_key = os.path.join(base_key_path, "scienceai-keys.json")
        if os.path.exists(target_key):
            try:
                with open(target_key) as file:
                    key_list = json.load(file)
                anthropic_available = bool(key_list.get("anthropic"))
            except (json.JSONDecodeError, OSError):
                pass

    # Also check for Claude on Vertex AI (GCP service account)
    if not anthropic_available:
        gcp_config = load_gcp_config("anthropic_vertex")
        if gcp_config:
            sa_path = gcp_config.get("service_account_path")
            if sa_path and os.path.exists(sa_path):
                anthropic_available = True

    available["anthropic"] = anthropic_available

    # Check Google
    google_available = bool(os.environ.get("GOOGLE_API_KEY")) or bool(os.environ.get("GEMINI_API_KEY"))
    if not google_available:
        base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
        target_key = os.path.join(base_key_path, "scienceai-keys.json")
        if os.path.exists(target_key):
            try:
                with open(target_key) as file:
                    key_list = json.load(file)
                google_available = bool(key_list.get("google")) or bool(key_list.get("gemini"))
            except (json.JSONDecodeError, OSError):
                pass

    # Also check for Gemini on Vertex AI (GCP service account)
    if not google_available:
        gcp_config = load_gcp_config("google_gcp")
        if gcp_config:
            sa_path = gcp_config.get("service_account_path")
            if sa_path and os.path.exists(sa_path):
                google_available = True

    available["google"] = google_available

    # Check Anthropic Vertex (Claude on GCP) - for internal tracking
    anthropic_vertex_available = False
    gcp_config = load_gcp_config("anthropic_vertex")
    if gcp_config:
        # Verify the service account file still exists
        sa_path = gcp_config.get("service_account_path")
        if sa_path and os.path.exists(sa_path):
            anthropic_vertex_available = True
    available["anthropic-vertex"] = anthropic_vertex_available

    return available


def get_current_provider_name() -> str:
    """Get the name of the currently configured provider."""
    return load_config().provider.value


def switch_provider(provider_name: str) -> bool:
    """Switch to a different LLM provider.

    Args:
        provider_name: Name of the provider ('openai', 'anthropic', 'google')

    Returns:
        True if switch was successful, False otherwise
    """
    global _config, _provider

    try:
        provider = Provider(provider_name.lower())
    except ValueError:
        return False

    # Check if the provider has an API key available
    available = get_available_providers()
    if not available.get(provider_name.lower(), False):
        return False

    # Create new config with the selected provider
    _config = LLMConfig(provider=provider)
    _provider = None  # Reset provider to force re-initialization

    # Persist the selection so it becomes the default
    save_config(_config)

    return True


def save_api_key(provider_name: str, api_key: str) -> bool:
    """Save an API key to the config file.

    Args:
        provider_name: Name of the provider ('openai', 'anthropic', 'google')
        api_key: The API key to save

    Returns:
        True if successful, False otherwise
    """
    base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
    os.makedirs(base_key_path, exist_ok=True)
    target_key = os.path.join(base_key_path, "scienceai-keys.json")

    # Load existing keys
    key_list = {}
    if os.path.exists(target_key):
        try:
            with open(target_key) as file:
                key_list = json.load(file)
        except (json.JSONDecodeError, OSError):
            pass

    # Update the key
    key_list[provider_name.lower()] = api_key

    # Save back
    try:
        with open(target_key, "w") as file:
            json.dump(key_list, file, indent=2)
        return True
    except OSError:
        return False


def save_gcp_config(
    service_account_path: str,
    project_id: str,
    region: str,
    use_for_gemini: bool = True,
    use_for_claude: bool = False,
) -> bool:
    """Save GCP service account configuration.

    Args:
        service_account_path: Path to the service account JSON file
        project_id: GCP project ID
        region: Vertex AI region (e.g., 'us-east5')
        use_for_gemini: Whether to use for Google Gemini models
        use_for_claude: Whether to use for Claude on Vertex AI

    Returns:
        True if successful, False otherwise
    """
    base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
    os.makedirs(base_key_path, exist_ok=True)
    target_key = os.path.join(base_key_path, "scienceai-keys.json")

    # Load existing keys
    key_list = {}
    if os.path.exists(target_key):
        try:
            with open(target_key) as file:
                key_list = json.load(file)
        except (json.JSONDecodeError, OSError):
            pass

    # Store GCP configuration
    gcp_config = {
        "service_account_path": service_account_path,
        "project_id": project_id,
        "region": region,
    }

    if use_for_gemini:
        key_list["google_gcp"] = gcp_config

    if use_for_claude:
        key_list["anthropic_vertex"] = gcp_config

    # Save back
    try:
        with open(target_key, "w") as file:
            json.dump(key_list, file, indent=2)
        return True
    except OSError:
        return False


def load_gcp_config(provider_key: str) -> dict | None:
    """Load GCP configuration for a specific provider.

    Args:
        provider_key: 'google_gcp' or 'anthropic_vertex'

    Returns:
        Dict with service_account_path, project_id, region or None
    """
    base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
    target_key = os.path.join(base_key_path, "scienceai-keys.json")

    if not os.path.exists(target_key):
        return None

    try:
        with open(target_key) as file:
            key_list = json.load(file)
            return key_list.get(provider_key)  # type: ignore
    except (json.JSONDecodeError, OSError):
        return None


def remove_gcp_config(remove_gemini: bool = False, remove_claude: bool = False) -> bool:
    """Remove GCP service account configuration.

    Args:
        remove_gemini: Remove Gemini GCP config
        remove_claude: Remove Claude Vertex config

    Returns:
        True if successful, False otherwise
    """
    base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
    target_key = os.path.join(base_key_path, "scienceai-keys.json")

    if not os.path.exists(target_key):
        return False

    try:
        with open(target_key) as file:
            key_list = json.load(file)

        removed = []
        if remove_gemini and "google_gcp" in key_list:
            del key_list["google_gcp"]
            removed.append("Gemini")

        if remove_claude and "anthropic_vertex" in key_list:
            del key_list["anthropic_vertex"]
            removed.append("Claude on Vertex")

        if not removed:
            return False

        with open(target_key, "w") as file:
            json.dump(key_list, file, indent=2)

        return True
    except (json.JSONDecodeError, OSError):
        return False


def validate_api_key(provider_name: str) -> tuple[bool, str]:
    """Validate an API key by making a minimal API call.

    Args:
        provider_name: Name of the provider ('openai', 'anthropic', 'google')

    Returns:
        Tuple of (is_valid, error_message)
    """
    provider_name = provider_name.lower()

    # Check if key exists first
    available = get_available_providers()
    if not available.get(provider_name, False):
        return False, "No API key configured"

    try:
        if provider_name == "openai":
            return _validate_openai_key()
        elif provider_name == "anthropic":
            return _validate_anthropic_key()
        elif provider_name == "google":
            return _validate_google_key()
        else:
            return False, f"Unknown provider: {provider_name}"
    except Exception as e:
        return False, str(e)


def _validate_openai_key() -> tuple[bool, str]:
    """Validate OpenAI API key with a minimal call."""
    import importlib.util

    if importlib.util.find_spec("openai") is None:
        return False, "OpenAI library not installed"

    config = LLMConfig(provider=Provider.OPENAI)
    provider = OpenAIProvider(config)

    try:
        # Use a minimal completion request - just ask for 1 token
        provider.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o-mini",  # Cheapest model
            max_tokens=1,
        )
        return True, "OK"
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return False, "Invalid API key"
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            # Key is valid but rate limited - that's fine
            return True, "OK (rate limited)"
        else:
            return False, error_msg[:100]


def _validate_anthropic_key() -> tuple[bool, str]:
    """Validate Anthropic API key with a minimal call."""
    import importlib.util

    if importlib.util.find_spec("anthropic") is None:
        return False, "Anthropic library not installed. Install with: pip install anthropic"

    config = LLMConfig(provider=Provider.ANTHROPIC)
    provider = AnthropicProvider(config)

    try:
        # Use a minimal completion request
        provider.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-haiku-4-5",  # Cheapest model
            max_tokens=1,
        )
        return True, "OK"
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower() or "invalid" in error_msg.lower():
            return False, "Invalid API key"
        elif "rate" in error_msg.lower() or "overloaded" in error_msg.lower():
            return True, "OK (rate limited)"
        else:
            return False, error_msg[:100]


def _validate_google_key() -> tuple[bool, str]:
    """Validate Google API key with a minimal call."""
    import importlib.util

    if importlib.util.find_spec("google.genai") is None:
        return False, "Google GenAI library not installed. Install with: pip install google-genai"

    config = LLMConfig(provider=Provider.GOOGLE)
    provider = GoogleProvider(config)

    try:
        # Use a minimal completion request
        provider.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemini-2.5-flash-lite",  # Cheapest model
            max_tokens=1,
        )
        return True, "OK"
    except Exception as e:
        error_msg = str(e)
        if "api key" in error_msg.lower() or "invalid" in error_msg.lower() or "authentication" in error_msg.lower():
            return False, "Invalid API key"
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return True, "OK (rate limited)"
        else:
            return False, error_msg[:100]


def validate_all_configured_keys() -> dict[str, tuple[bool, str]]:
    """Validate all API keys that are configured.

    Returns:
        Dictionary mapping provider names to (is_valid, message) tuples
    """
    results = {}
    available = get_available_providers()

    for provider_name, is_available in available.items():
        # Skip GCP-based providers (they use service accounts, not API keys)
        if provider_name in ["google_gcp", "anthropic-vertex"]:
            if is_available:
                results[provider_name] = (True, "GCP service account configured")
            continue

        # Check if this provider is available via GCP (not API key)
        if provider_name == "anthropic" and available.get("anthropic-vertex"):
            # Available via Vertex AI
            results[provider_name] = (True, "Available via Vertex AI")
            continue
        elif provider_name == "google" and not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            # Check if available via GCP service account
            gcp_config = load_gcp_config("google_gcp")
            if gcp_config and os.path.exists(gcp_config.get("service_account_path", "")):
                results[provider_name] = (True, "Available via Vertex AI")
                continue

        if is_available:
            results[provider_name] = validate_api_key(provider_name)
        else:
            results[provider_name] = (False, "No API key")

    return results


def setup_api_keys_interactive() -> None:
    """Interactive CLI prompt to set up API keys."""
    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║              ScienceAI API Key Setup                      ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    providers = [
        ("openai", "OpenAI", "Get your key at: https://platform.openai.com/api-keys"),
        ("anthropic", "Anthropic (Claude)", "Get your key at: https://console.anthropic.com/"),
        ("google", "Google (Gemini)", "Get your key at: https://aistudio.google.com/apikey"),
    ]

    available = get_available_providers()

    for provider_id, provider_name, help_url in providers:
        current_status = "✓ configured" if available.get(provider_id) else "✗ not configured"
        print(f"\n{provider_name}: [{current_status}]")
        print(f"  {help_url}")

        response = input(f"  Enter API key for {provider_name} (or press Enter to skip): ").strip()

        if response:
            if save_api_key(provider_id, response):
                print(f"  ✓ Saved {provider_name} API key")
            else:
                print(f"  ✗ Failed to save {provider_name} API key")

    print("\n" + "═" * 60)
    print("Validating configured API keys...\n")

    results = validate_all_configured_keys()
    any_valid = False

    for provider_name, (is_valid, message) in results.items():
        status = "✓" if is_valid else "✗"
        print(f"  {status} {provider_name}: {message}")
        if is_valid:
            any_valid = True

    if not any_valid:
        print("\n⚠ No valid API keys configured. ScienceAI requires at least one valid API key.")
    else:
        print("\n✓ Setup complete!")
