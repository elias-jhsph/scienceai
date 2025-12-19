import os
import unittest
from unittest.mock import MagicMock, patch

from scienceai.llm_providers import AnthropicProvider, GoogleProvider, LLMConfig, OpenAIProvider, Provider


class TestTokenCounting(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "Hello world!"}, {"role": "assistant", "content": "Hi there!"}]

    def test_openai_token_counting(self):
        config = LLMConfig(provider=Provider.OPENAI, api_key="test-key")
        # Mock OpenAI client creation
        with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
            provider = OpenAIProvider(config)
            count = provider.count_tokens(self.messages)
            print(f"OpenAI count: {count}")
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

    def test_anthropic_token_counting_mock(self):
        config = LLMConfig(provider=Provider.ANTHROPIC, api_key="sk-ant-test")

        # Patch load_gcp_config to return None -> Direct Client
        with (
            patch("scienceai.llm_providers.load_gcp_config", return_value=None),
            patch("anthropic.Anthropic") as MockAnthropic,
        ):
            mock_client = MockAnthropic.return_value
            mock_client.beta.messages.count_tokens.return_value.input_tokens = 42

            provider = AnthropicProvider(config)
            count = provider.count_tokens(self.messages)
            self.assertEqual(count, 42)

    def test_anthropic_vertex_token_counting_mock(self):
        """Test Anthropic Vertex token counting using REST API."""
        config = LLMConfig(provider=Provider.ANTHROPIC)
        gcp_config = {"service_account_path": "/tmp/dummy.json", "project_id": "test-project", "region": "us-east5"}

        with (
            patch("scienceai.llm_providers.load_gcp_config", return_value=gcp_config),
            patch("scienceai.llm_providers.os.path.exists", return_value=True),
            patch.dict(os.environ, {}, clear=True),
            patch("scienceai.llm_providers.AnthropicProvider._init_vertex_client") as _,
        ):
            provider = AnthropicProvider(config)

            # Manually set up Vertex attributes that _init_vertex_client would set
            provider._project_id = "test-project"
            provider._region = "us-east5"
            provider._sa_path = "/tmp/dummy.json"
            provider.client = MagicMock()  # Just a mock client, no spec needed

            with (
                patch("requests.post") as mock_post,
                patch("google.auth.default", return_value=(MagicMock(), "proj")),
                patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_creds,
            ):
                # Set up credentials mock
                mock_cred_instance = MagicMock()
                mock_cred_instance.token = "fake-token"
                mock_creds.return_value = mock_cred_instance

                # Set up response mock
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"input_tokens": 123}
                mock_post.return_value = mock_response

                count = provider.count_tokens(self.messages)
                self.assertEqual(count, 123)

                # Verify URL uses us-east5
                args, _ = mock_post.call_args
                self.assertIn("us-east5-aiplatform.googleapis.com", args[0])

    def test_google_token_counting_mock(self):
        config = LLMConfig(provider=Provider.GOOGLE, api_key="gemini-test-key")

        with (
            patch("scienceai.llm_providers.load_gcp_config", return_value=None),
            patch("google.genai.Client") as MockGenAI,
        ):
            mock_client = MockGenAI.return_value
            mock_client.models.count_tokens.return_value.total_tokens = 15

            provider = GoogleProvider(config)
            count = provider.count_tokens(self.messages)
            self.assertEqual(count, 15)

    def test_google_vertex_token_counting_mock(self):
        """Test Google Vertex token counting."""
        config = LLMConfig(provider=Provider.GOOGLE)
        gcp_config = {"service_account_path": "/tmp/dummy.json", "project_id": "test-project", "region": "global"}

        # Mock google.genai before GoogleProvider tries to import it
        with (
            patch("scienceai.llm_providers.load_gcp_config", return_value=gcp_config),
            patch("scienceai.llm_providers.os.path.exists", return_value=True),
            patch("scienceai.llm_providers.GoogleProvider._init_vertex_client") as _,
        ):
            # Import and patch google.genai after the provider is created but before it uses it
            import sys

            mock_genai_module = MagicMock()
            mock_client = MagicMock()
            mock_client.models.count_tokens.return_value.total_tokens = 99
            mock_genai_module.Client.return_value = mock_client
            sys.modules["google"] = MagicMock()
            sys.modules["google.genai"] = mock_genai_module
            provider = GoogleProvider(config)
            # Mimic Vertex setup - must set all attributes that _init_vertex_client would set
            provider._use_vertex = True
            provider.credentials = MagicMock()  # Credentials for Google auth
            provider.project_id = "test-project"  # Required for Vertex token counting
            provider.region = "global"  # Required for Vertex token counting

            # Set the provider's client to the mock
            provider.client = mock_client

            count = provider.count_tokens(self.messages)
            self.assertEqual(count, 99)


if __name__ == "__main__":
    unittest.main()
