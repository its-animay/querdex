from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic LLM interface.

    All methods return raw text.  Callers that need structured output
    should pass the response through ``validate_llm_payload()``.
    """

    tier1_model: str
    tier2_model: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        """Send a completion request and return the response text."""
        ...


def build_llm_client() -> LLMClient | None:
    """Build an LLMClient from environment variables.

    Environment variables:
        QUERDEX_LLM_PROVIDER   : ``anthropic`` or ``openai``
        QUERDEX_LLM_API_KEY    : provider API key
        QUERDEX_LLM_TIER1_MODEL: cheap/fast model identifier
        QUERDEX_LLM_TIER2_MODEL: powerful model identifier

    Returns ``None`` when ``QUERDEX_LLM_PROVIDER`` is not set,
    which causes all callers to fall back to their heuristic paths.
    """
    provider = os.getenv("QUERDEX_LLM_PROVIDER", "").strip().lower()
    api_key = os.getenv("QUERDEX_LLM_API_KEY", "").strip()

    if not provider:
        return None

    if provider == "anthropic":
        from querdex.llm.anthropic_client import AnthropicLLMClient

        tier1 = os.getenv("QUERDEX_LLM_TIER1_MODEL", "claude-haiku-4-5-20251001")
        tier2 = os.getenv("QUERDEX_LLM_TIER2_MODEL", "claude-sonnet-4-6")
        return AnthropicLLMClient(api_key=api_key or None, tier1_model=tier1, tier2_model=tier2)

    if provider == "openai":
        from querdex.llm.openai_client import OpenAILLMClient

        tier1 = os.getenv("QUERDEX_LLM_TIER1_MODEL", "gpt-4o-mini")
        tier2 = os.getenv("QUERDEX_LLM_TIER2_MODEL", "gpt-4o")
        return OpenAILLMClient(api_key=api_key or None, tier1_model=tier1, tier2_model=tier2)

    msg = f"Unknown QUERDEX_LLM_PROVIDER: {provider!r}. Choose 'anthropic' or 'openai'."
    raise ValueError(msg)
