from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"```\s*$", re.MULTILINE)


def extract_json(raw: str) -> str:
    """Extract a JSON object or array from a raw LLM response.

    Handles the three common formats OpenAI / Anthropic models produce:
      1. Bare JSON:           ``{"key": "value"}``
      2. Markdown code block: `` ```json\\n{...}\\n``` ``
      3. Preamble text:       ``Here is the result: {"key": "value"}``
    """
    text = _CODE_FENCE_RE.sub("", raw.strip())
    text = _FENCE_CLOSE_RE.sub("", text).strip()

    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace == -1 and first_bracket == -1:
        return text  # let the caller's json.loads raise with a clear message

    if first_brace == -1:
        return text[first_bracket:]
    if first_bracket == -1:
        return text[first_brace:]

    return text[min(first_brace, first_bracket):]


class LLMValidationError(ValueError):
    """Raised when an LLM payload does not conform to a schema."""


def validate_llm_payload(model: type[T], payload: str | dict[str, object]) -> T:
    data: dict[str, object]
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMValidationError("LLM payload is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMValidationError("LLM payload must decode to an object")
        data = parsed
    else:
        data = payload

    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise LLMValidationError("LLM payload failed schema validation") from exc
