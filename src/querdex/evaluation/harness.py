from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from querdex.schemas import QueryResult


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_keywords: list[str]
    intent: str


@dataclass(frozen=True)
class EvalSummary:
    total_cases: int
    intent_accuracy: float
    citation_coverage: float
    avg_latency_ms: float
    cache_hit_rate: float


class EvaluationHarness:
    def load_cases(self, path: str | Path) -> list[EvalCase]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [
            EvalCase(
                query=str(item["query"]),
                expected_keywords=[str(v) for v in item.get("expected_keywords", [])],
                intent=str(item.get("intent", "single_doc")),
            )
            for item in data
        ]

    def score_results(self, cases: list[EvalCase], results: list[QueryResult]) -> EvalSummary:
        if len(cases) != len(results):
            msg = "cases/results length mismatch"
            raise ValueError(msg)

        intent_hits = sum(1 for case, result in zip(cases, results, strict=True) if result.intent_type == case.intent)
        citation_hits = sum(1 for result in results if result.source_nodes)
        latencies = [result.latency_ms for result in results]
        cache_hits = sum(1 for result in results if result.cache_hit)

        return EvalSummary(
            total_cases=len(cases),
            intent_accuracy=intent_hits / max(1, len(cases)),
            citation_coverage=citation_hits / max(1, len(cases)),
            avg_latency_ms=mean(latencies) if latencies else 0.0,
            cache_hit_rate=cache_hits / max(1, len(cases)),
        )
