from __future__ import annotations


class FakeLLMClient:
    """Deterministic LLM stub for unit tests.

    Usage::

        fake = FakeLLMClient(responses={"summary": '{"summary": "Test summary."}'})
        builder = AdaptiveTreeBuilder(llm_client=fake)

    The ``responses`` dict maps a *substring* of the ``user`` prompt to the
    reply string.  The first matching key wins.  If no key matches the
    ``default`` is returned.
    """

    tier1_model: str = "fake-tier1"
    tier2_model: str = "fake-tier2"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = '{"summary": "Fake LLM summary."}',
    ) -> None:
        self.responses: dict[str, str] = responses or {}
        self.default = default
        self.calls: list[dict[str, str]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append({"system": system, "user": user, "model": model})
        for key, reply in self.responses.items():
            if key in user:
                return reply
        return self.default
