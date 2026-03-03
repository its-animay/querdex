from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from querdex.evaluation.harness import EvalSummary
from querdex.schemas import QueryResult


@dataclass(frozen=True)
class CostModel:
    tier1_call_cost_usd: float = 0.0001
    tier2_call_cost_usd: float = 0.0006


@dataclass(frozen=True)
class RetrievalMetricsSummary:
    total_queries: int
    avg_latency_ms: float
    p95_latency_ms: float
    avg_cost_usd: float
    cache_hit_rate: float


class RetrievalMetricsHarness:
    def summarize(self, results: list[QueryResult], cost_model: CostModel | None = None) -> RetrievalMetricsSummary:
        model = cost_model or CostModel()
        if not results:
            return RetrievalMetricsSummary(
                total_queries=0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                avg_cost_usd=0.0,
                cache_hit_rate=0.0,
            )

        latencies = sorted(result.latency_ms for result in results)
        costs = [self._estimate_cost(result, model) for result in results]
        cache_hits = sum(1 for result in results if result.cache_hit)
        p95_idx = int(0.95 * (len(latencies) - 1))

        return RetrievalMetricsSummary(
            total_queries=len(results),
            avg_latency_ms=mean(latencies),
            p95_latency_ms=float(latencies[p95_idx]),
            avg_cost_usd=mean(costs),
            cache_hit_rate=cache_hits / len(results),
        )

    @staticmethod
    def _estimate_cost(result: QueryResult, model: CostModel) -> float:
        return (
            (result.tier1_calls * model.tier1_call_cost_usd)
            + (result.tier2_calls * model.tier2_call_cost_usd)
        )


class KPIDashboardBuilder:
    def build(
        self,
        *,
        eval_summary: EvalSummary,
        retrieval_summary: RetrievalMetricsSummary,
    ) -> dict[str, float | int]:
        return {
            "total_eval_cases": eval_summary.total_cases,
            "intent_accuracy": round(eval_summary.intent_accuracy, 4),
            "citation_coverage": round(eval_summary.citation_coverage, 4),
            "avg_latency_ms": round(retrieval_summary.avg_latency_ms, 2),
            "p95_latency_ms": round(retrieval_summary.p95_latency_ms, 2),
            "avg_cost_usd": round(retrieval_summary.avg_cost_usd, 6),
            "cache_hit_rate": round(retrieval_summary.cache_hit_rate, 4),
        }

    @staticmethod
    def write_json(path: str | Path, payload: dict[str, float | int]) -> None:
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
