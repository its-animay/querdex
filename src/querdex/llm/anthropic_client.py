from __future__ import annotations


class AnthropicLLMClient:
    """LLMClient backed by the Anthropic Messages API.

    Requires the ``anthropic`` optional extra:
        pip install querdex[anthropic]
        uv sync --extra anthropic
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        tier1_model: str = "claude-haiku-4-5-20251001",
        tier2_model: str = "claude-sonnet-4-6",
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            msg = (
                "The 'anthropic' package is required to use AnthropicLLMClient.\n"
                "Install it with:  pip install querdex[anthropic]"
            )
            raise ImportError(msg) from exc

        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self._client = _anthropic.Anthropic(api_key=api_key) if api_key else _anthropic.Anthropic()

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        block = message.content[0]
        return block.text if hasattr(block, "text") else str(block)
