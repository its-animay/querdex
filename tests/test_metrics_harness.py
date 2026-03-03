from __future__ import annotations

import json
from pathlib import Path

from querdex.evaluation import (
    CostModel,
    EvalSummary,
    KPIDashboardBuilder,
    RetrievalMetricsHarness,
)
from querdex.schemas import QueryResult


def _result(
    *,
    query_id: str,
    latency_ms: int,
    tier1_calls: int,
    tier2_calls: int,
    cache_hit: bool,
) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        original_query="q",
        rewritten_query="q",
        intent_type="single_doc",
        traversal_path=[],
        source_nodes=[],
        answer="ok",
        confidence=0.5,
        latency_ms=latency_ms,
        tier1_calls=tier1_calls,
        tier2_calls=tier2_calls,
        cache_hit=cache_hit,
    )


def test_retrieval_metrics_harness_summarizes_latency_cost_and_cache() -> None:
    harness = RetrievalMetricsHarness()
    summary = harness.summarize(
        [
            _result(query_id="q1", latency_ms=100, tier1_calls=1, tier2_calls=2, cache_hit=False),
            _result(query_id="q2", latency_ms=200, tier1_calls=1, tier2_calls=1, cache_hit=True),
            _result(query_id="q3", latency_ms=300, tier1_calls=0, tier2_calls=3, cache_hit=False),
        ],
        cost_model=CostModel(tier1_call_cost_usd=0.001, tier2_call_cost_usd=0.002),
    )

    assert summary.total_queries == 3
    assert summary.avg_latency_ms > 0
    assert summary.p95_latency_ms >= 200
    assert summary.avg_cost_usd > 0
    assert 0 < summary.cache_hit_rate < 1


def test_kpi_dashboard_builder_outputs_expected_keys() -> None:
    dashboard = KPIDashboardBuilder().build(
        eval_summary=EvalSummary(
            total_cases=3,
            intent_accuracy=1.0,
            citation_coverage=0.66,
            avg_latency_ms=120.0,
            cache_hit_rate=0.33,
        ),
        retrieval_summary=RetrievalMetricsHarness().summarize(
            [
                _result(query_id="q1", latency_ms=120, tier1_calls=1, tier2_calls=1, cache_hit=False),
            ]
        ),
    )

    assert dashboard["total_eval_cases"] == 3
    assert "p95_latency_ms" in dashboard
    assert "avg_cost_usd" in dashboard


def test_kpi_baseline_fixture_has_required_metrics() -> None:
    baseline = json.loads(Path("tests/fixtures/eval/kpi_baseline.json").read_text(encoding="utf-8"))
    required = {"intent_accuracy", "citation_coverage", "p95_latency_ms", "avg_cost_usd", "cache_hit_rate"}
    assert required.issubset(set(baseline))
