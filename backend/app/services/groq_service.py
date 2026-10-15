from dataclasses import dataclass
import logging
from typing import Optional
from groq import (
    Groq,
    GroqError,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)
from app.core.config import settings

logger = logging.getLogger("sentinel.groq")


@dataclass
class ProviderResult:
    """Structured result from the AI provider distinguishing valid output from failures."""
    content: Optional[str] = None
    error: Optional[str] = None
    success: bool = True


class GroqLLMService:
    """
    Service wrapper for the official Groq Python SDK.
    Enforces that only policy-cleared, sanitized prompts are forwarded to Groq.
    Distinguishes successful model output from provider failures.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    @property
    def api_key(self) -> str:
        return self._api_key if self._api_key is not None else settings.GROQ_API_KEY

    @property
    def model(self) -> str:
        return self._model or settings.GROQ_MODEL

    @property
    def timeout(self) -> float:
        return (
            self._timeout
            if self._timeout is not None
            else settings.GROQ_TIMEOUT_SECONDS
        )

    def _get_client(self) -> Groq:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured on the server.")
        return Groq(api_key=self.api_key, timeout=self.timeout)

    def generate_response(self, sanitized_prompt: str) -> ProviderResult:
        """
        Sends a sanitized prompt to Groq for inference.
        Returns a ProviderResult with content on success or error details on failure.
        Guarantees that raw exceptions, internal traces, and API keys are never exposed.
        """
        if not sanitized_prompt or not sanitized_prompt.strip():
            return ProviderResult(
                content=None,
                error="[AI Provider Error: Received empty sanitized prompt.]",
                success=False,
            )

        try:
            client = self._get_client()
            chat_completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant accessed through the SentinelAI Security Gateway. "
                            "You process strictly sanitized, policy-cleared prompts. "
                            "Respect redactions (e.g., [REDACTED_...]) and provide helpful, direct responses."
                        ),
                    },
                    {
                        "role": "user",
                        "content": sanitized_prompt,
                    },
                ],
                temperature=0.5,
            )

            if not chat_completion.choices:
                return ProviderResult(
                    content=None,
                    error="[AI Provider Error: No response choices returned by Groq.]",
                    success=False,
                )

            first_choice = chat_completion.choices[0]
            content = first_choice.message.content

            if content is None or not content.strip():
                return ProviderResult(
                    content=None,
                    error="[AI Provider Error: Received empty completion from Groq.]",
                    success=False,
                )

            return ProviderResult(content=content.strip(), error=None, success=True)

        except AuthenticationError:
            logger.error("Groq authentication failed. Please verify GROQ_API_KEY.")
            return ProviderResult(
                content=None,
                error="[AI Provider Auth Error: Authentication failed. Please verify server API key configuration.]",
                success=False,
            )

        except RateLimitError:
            logger.warning("Groq rate limit encountered.")
            return ProviderResult(
                content=None,
                error="[AI Provider Rate Limit: Upstream rate limit reached. Please retry in a few moments.]",
                success=False,
            )

        except APITimeoutError:
            logger.warning("Groq request timed out after %s seconds.", self.timeout)
            return ProviderResult(
                content=None,
                error=f"[AI Provider Timeout: Request exceeded the {self.timeout}s timeout limit.]",
                success=False,
            )

        except APIConnectionError:
            logger.warning("Groq connection failed.")
            return ProviderResult(
                content=None,
                error="[AI Provider Network Error: Unable to establish connection to Groq API.]",
                success=False,
            )

        except APIStatusError as e:
            logger.warning(
                "Groq API returned an error status code: %s",
                getattr(e, "status_code", "unknown"),
            )
            return ProviderResult(
                content=None,
                error="[AI Provider Error: Upstream inference service returned an error status.]",
                success=False,
            )

        except ValueError as e:
            logger.warning("Configuration error: %s", str(e))
            return ProviderResult(
                content=None,
                error=f"[AI Provider Config Error: {str(e)}]",
                success=False,
            )

        except GroqError as e:
            logger.warning("Groq SDK error encountered: %s", type(e).__name__)
            return ProviderResult(
                content=None,
                error="[AI Provider Error: An unexpected error occurred while communicating with Groq.]",
                success=False,
            )

        except Exception as e:
            logger.error("Unexpected error in Groq service: %s", type(e).__name__)
            return ProviderResult(
                content=None,
                error="[AI Provider Error: An unexpected system error occurred during AI processing.]",
                success=False,
            )


groq_service = GroqLLMService()
