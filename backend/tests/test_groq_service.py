import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groq import (
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)
import httpx

from app.core.config import settings
from app.services.groq_service import GroqLLMService, ProviderResult
from app.services.llm_provider import get_llm_response, LLMResult


class TestGroqLLMService(unittest.TestCase):

    def setUp(self):
        self.mock_request = httpx.Request(
            "POST", "https://api.groq.com/openai/v1/chat/completions"
        )
        self.mock_response = httpx.Response(
            status_code=400,
            request=self.mock_request,
        )

    def test_mock_mode_default(self):
        """When USE_MOCK_LLM is True, dispatcher returns mock AI response without network calls."""
        with patch.object(settings, "USE_MOCK_LLM", True):
            result: LLMResult = get_llm_response("Please summarize this report.")
            self.assertTrue(result.success)
            self.assertIn("[Mock AI]", result.content)
            self.assertIsNone(result.error)

    def test_groq_service_success(self):
        """Groq service successfully produces text when upstream call succeeds."""
        mock_choice = MagicMock()
        mock_choice.message.content = "Analysis of the sanitized data is complete."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        service = GroqLLMService(
            api_key="mock_key_for_test",
            model="llama-3.3-70b-versatile",
            timeout=5.0,
        )

        with patch.object(service, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_get_client.return_value = mock_client

            result: ProviderResult = service.generate_response("Sanitized query text")
            self.assertTrue(result.success)
            self.assertEqual(
                result.content, "Analysis of the sanitized data is complete."
            )
            self.assertIsNone(result.error)

            # Verify prompt and model were passed
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(call_kwargs["model"], "llama-3.3-70b-versatile")
            user_msg = [m for m in call_kwargs["messages"] if m["role"] == "user"][0]
            self.assertEqual(user_msg["content"], "Sanitized query text")

    def test_groq_service_missing_api_key(self):
        """Groq service returns a configuration error when no API key is set."""
        service = GroqLLMService(api_key="")
        result: ProviderResult = service.generate_response("Some sanitized text")
        self.assertFalse(result.success)
        self.assertIsNone(result.content)
        self.assertIn("[AI Provider Config Error:", result.error)

    def test_groq_service_auth_error(self):
        """Groq service safely catches AuthenticationError and does not leak keys."""
        service = GroqLLMService(api_key="invalid_key")

        with patch.object(service, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = AuthenticationError(
                message="Invalid API Key",
                response=self.mock_response,
                body={},
            )
            mock_get_client.return_value = mock_client

            result: ProviderResult = service.generate_response("Some query")
            self.assertFalse(result.success)
            self.assertIsNone(result.content)
            self.assertIn("[AI Provider Auth Error:", result.error)
            self.assertNotIn("invalid_key", result.error)

    def test_groq_service_rate_limit(self):
        """Groq service safely catches RateLimitError and returns clear user message."""
        service = GroqLLMService(api_key="mock_key")

        with patch.object(service, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=self.mock_response,
                body={},
            )
            mock_get_client.return_value = mock_client

            result: ProviderResult = service.generate_response("Some query")
            self.assertFalse(result.success)
            self.assertIsNone(result.content)
            self.assertIn("[AI Provider Rate Limit:", result.error)

    def test_groq_service_timeout(self):
        """Groq service safely catches APITimeoutError."""
        service = GroqLLMService(api_key="mock_key", timeout=10.0)

        with patch.object(service, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APITimeoutError(
                request=self.mock_request
            )
            mock_get_client.return_value = mock_client

            result: ProviderResult = service.generate_response("Some query")
            self.assertFalse(result.success)
            self.assertIsNone(result.content)
            self.assertIn("[AI Provider Timeout:", result.error)
            self.assertIn("10.0s", result.error)

    def test_groq_service_network_error(self):
        """Groq service safely catches APIConnectionError."""
        service = GroqLLMService(api_key="mock_key")

        with patch.object(service, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=self.mock_request
            )
            mock_get_client.return_value = mock_client

            result: ProviderResult = service.generate_response("Some query")
            self.assertFalse(result.success)
            self.assertIsNone(result.content)
            self.assertIn("[AI Provider Network Error:", result.error)

    def test_groq_service_empty_completion(self):
        """Groq service safely handles empty choices or null content."""
        mock_completion = MagicMock()
        mock_completion.choices = []

        service = GroqLLMService(api_key="mock_key")

        with patch.object(service, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_get_client.return_value = mock_client

            result: ProviderResult = service.generate_response("Some query")
            self.assertFalse(result.success)
            self.assertIsNone(result.content)
            self.assertIn(
                "[AI Provider Error: No response choices returned", result.error
            )

    def test_llm_provider_dispatcher_groq_mode(self):
        """Dispatcher calls Groq when USE_MOCK_LLM is False and GROQ_API_KEY is present."""
        with patch.object(settings, "USE_MOCK_LLM", False), \
             patch.object(settings, "LLM_PROVIDER", "groq"), \
             patch.object(settings, "GROQ_API_KEY", "mock_valid_key"), \
             patch("app.services.llm_provider.groq_service.generate_response") as mock_groq_call:

            mock_groq_call.return_value = ProviderResult(
                content="Groq response from dispatcher",
                error=None,
                success=True,
            )
            result: LLMResult = get_llm_response("Sanitized query")
            self.assertTrue(result.success)
            self.assertEqual(result.content, "Groq response from dispatcher")
            self.assertIsNone(result.error)
            mock_groq_call.assert_called_once_with("Sanitized query")

    def test_llm_provider_dispatcher_missing_key(self):
        """Dispatcher returns configuration error if Groq selected but no key configured."""
        with patch.object(settings, "USE_MOCK_LLM", False), \
             patch.object(settings, "LLM_PROVIDER", "groq"), \
             patch.object(settings, "GROQ_API_KEY", ""):

            result: LLMResult = get_llm_response("Sanitized query")
            self.assertFalse(result.success)
            self.assertIsNone(result.content)
            self.assertIn("[AI Provider Config Error:", result.error)


if __name__ == "__main__":
    unittest.main()
