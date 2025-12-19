import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Mock anthropic module before import
mock_anthropic = MagicMock()
sys.modules["anthropic"] = mock_anthropic

from scienceai.llm_providers import AnthropicProvider, LLMConfig, Provider


class TestTokenClamping(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        # Mock both messages.create and beta.messages.create
        self.mock_response = MagicMock()
        self.mock_response.content = [MagicMock(text="test response")]
        self.mock_client.messages.create.return_value = self.mock_response
        self.mock_client.beta.messages.create.return_value = self.mock_response

        mock_anthropic.Anthropic.return_value = self.mock_client

        # Also mock AnthropicVertex
        mock_anthropic.AnthropicVertex.return_value = self.mock_client

    @patch("scienceai.llm_providers.load_gcp_config", return_value=None)
    def test_clamping_with_restricted_max_tokens(self, mock_load_config):
        """Test that high reasoning budget is clamped when max_tokens is restrictive."""
        config = LLMConfig(provider=Provider.ANTHROPIC, api_key="test-key")
        provider = AnthropicProvider(config)

        # Test: High reasoning (16384) vs Restricted Max Tokens (5000)
        # Logic: budget = 5000 - 2048 = 2952
        provider.chat_completion(messages=[{"role": "user", "content": "hi"}], reasoning_effort="high", max_tokens=5000)

        # When thinking is enabled, it uses beta.messages.create
        call_kwargs = self.mock_client.beta.messages.create.call_args.kwargs
        thinking = call_kwargs.get("thinking")
        self.assertIsNotNone(thinking)
        print(f"Restricted Max Tokens Test: clamped budget to {thinking['budget_tokens']}")
        self.assertEqual(thinking["budget_tokens"], 2952)

    @patch("scienceai.llm_providers.load_gcp_config", return_value=None)
    def test_high_reasoning_no_clamp_default_max_tokens(self, mock_load_config):
        """Test that high reasoning is NOT clamped when default max_tokens (20000) is used."""
        config = LLMConfig(provider=Provider.ANTHROPIC, api_key="test-key")
        provider = AnthropicProvider(config)

        # Test: High reasoning (16384) vs Default Max Tokens (20000)
        # Logic: 16384 < 20000, no clamp needed
        provider.chat_completion(messages=[{"role": "user", "content": "hi"}], reasoning_effort="high")

        # Verify - when thinking is enabled, it uses beta.messages.create
        call_kwargs = self.mock_client.beta.messages.create.call_args.kwargs
        thinking = call_kwargs.get("thinking")
        self.assertIsNotNone(thinking)
        print(f"Default Max Tokens Test: budget is {thinking['budget_tokens']}")
        # High budget (16384) is less than default max_tokens (20000), so no clamping
        self.assertEqual(thinking["budget_tokens"], 16384)
        self.assertEqual(thinking["type"], "enabled")

    @patch("scienceai.llm_providers.load_gcp_config", return_value=None)
    def test_medium_reasoning_no_clamp(self, mock_load_config):
        """Test that medium reasoning is NOT clamped with default max_tokens."""
        config = LLMConfig(provider=Provider.ANTHROPIC, api_key="test-key")
        provider = AnthropicProvider(config)

        # Test: Medium reasoning (8192) vs Default Max Tokens (20000)
        # Logic: 8192 < 20000, no clamp needed
        provider.chat_completion(messages=[{"role": "user", "content": "hi"}], reasoning_effort="medium")

        # When thinking is enabled, it uses beta.messages.create
        call_kwargs = self.mock_client.beta.messages.create.call_args.kwargs
        thinking = call_kwargs.get("thinking")
        self.assertIsNotNone(thinking)
        print(f"No Clamp Test: budget is {thinking['budget_tokens']}")
        self.assertEqual(thinking["budget_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
