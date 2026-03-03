from __future__ import annotations

import re
from typing import Literal

from querdex.schemas import QueryAnalysis
from querdex.utils.query_cluster import compute_query_cluster_id

_ENTITY_RE = re.compile(
    r"\b(?:Table\s+[A-Za-z0-9\.\-]+|Section\s+[A-Za-z0-9\.\-]+|[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)"
)
_ENTITY_STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "who",
    "why",
    "how",
    "this",
    "that",
    "these",
    "those",
}


class QueryAnalyzer:
    def analyze(self, query: str, session_turns: list[dict[str, str]] | None = None) -> QueryAnalysis:
        turns = session_turns or []
        rewritten = self._rewrite(query, turns)
        intent = self._classify(rewritten)
        entities = sorted(
            {
                value
                for match in _ENTITY_RE.finditer(rewritten)
                for value in [match.group(0).strip()]
                if value and value.lower() not in _ENTITY_STOPWORDS
            }
        )
        cluster_id = compute_query_cluster_id(rewritten)
        return QueryAnalysis(
            intent_type=intent,
            extracted_entities=entities,
            rewritten_query=rewritten,
            query_cluster_id=cluster_id,
        )

    @staticmethod
    def _classify(query: str) -> Literal["single_doc", "multi_doc", "graph"]:
        q = query.lower()
        if any(token in q for token in ["compare", "vs", "versus", "difference between"]):
            return "multi_doc"
        if any(token in q for token in ["refer", "references", "cite", "what cites", "connected"]):
            return "graph"
        return "single_doc"

    @staticmethod
    def _rewrite(query: str, turns: list[dict[str, str]]) -> str:
        stripped = query.strip()
        if not turns:
            return stripped

        lower = stripped.lower()
        if any(lower.startswith(prefix) for prefix in ["what about", "how about", "and ", "it ", "that "]):
            last = turns[0]
            anchor = last.get("rewritten_query", "").strip()
            if anchor:
                return f"{stripped} Context from previous turn: {anchor}"
        return stripped
