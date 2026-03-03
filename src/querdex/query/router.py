from __future__ import annotations

from querdex.schemas import QueryAnalysis


class QueryRouter:
    def route(self, analysis: QueryAnalysis) -> str:
        return analysis.intent_type
