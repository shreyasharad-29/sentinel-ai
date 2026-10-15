from dataclasses import dataclass
from typing import Optional
from app.core.config import settings
from app.services.mock_llm import mock_llm_service
from app.services.groq_service import groq_service, ProviderResult


@dataclass
class LLMResult:
    """Structured LLM result clearly separating successful model content from provider failure."""
    content: Optional[str] = None
    error: Optional[str] = None
    success: bool = True


def get_llm_response(sanitized_prompt: str) -> LLMResult:
    """
    Unified LLM provider dispatcher:
    1. Retains mock mode as default when USE_MOCK_LLM is True or LLM_PROVIDER is 'mock'.
    2. Routes to Groq when USE_MOCK_LLM is False and LLM_PROVIDER is 'groq'.
    3. Handles missing API keys cleanly without crashing or exposing secrets.
    4. Distinguishes successful text completion from provider failure.
    """
    if settings.USE_MOCK_LLM or settings.LLM_PROVIDER == "mock":
        content = mock_llm_service.generate_response(sanitized_prompt)
        return LLMResult(content=content, error=None, success=True)

    if settings.LLM_PROVIDER == "groq":
        if not settings.GROQ_API_KEY:
            return LLMResult(
                content=None,
                error="[AI Provider Config Error: GROQ_API_KEY is not configured on the server.]",
                success=False,
            )
        result: ProviderResult = groq_service.generate_response(sanitized_prompt)
        return LLMResult(
            content=result.content,
            error=result.error,
            success=result.success,
        )

    content = mock_llm_service.generate_response(sanitized_prompt)
    return LLMResult(content=content, error=None, success=True)
