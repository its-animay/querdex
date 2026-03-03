from __future__ import annotations


class OpenAILLMClient:
    """LLMClient backed by the OpenAI Chat Completions API.

    Requires the ``openai`` optional extra:
        pip install querdex[openai]
        uv sync --extra openai
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        tier1_model: str = "gpt-4o-mini",
        tier2_model: str = "gpt-4o",
    ) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            msg = (
                "The 'openai' package is required to use OpenAILLMClient.\n"
                "Install it with:  pip install querdex[openai]"
            )
            raise ImportError(msg) from exc

        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self._client = _openai.OpenAI(api_key=api_key) if api_key else _openai.OpenAI()

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
