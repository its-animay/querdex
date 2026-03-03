from __future__ import annotations

from pathlib import Path

from querdex.evaluation import EvaluationHarness
from querdex.schemas import QueryResult, SourceNode


def test_evaluation_harness_loads_fixture_and_scores() -> None:
    fixture = Path("tests/fixtures/eval/baseline_cases.json")
    harness = EvaluationHarness()
    cases = harness.load_cases(fixture)

    assert len(cases) == 3
    results = [
        QueryResult(
            query_id="q1",
            original_query=cases[0].query,
            rewritten_query=cases[0].query,
            intent_type="single_doc",
            traversal_path=["n1"],
            source_nodes=[
                SourceNode(
                    node_id="n1",
                    doc_id="doc1",
                    title="Revenue",
                    pages="1-1",
                    confidence=0.9,
                )
            ],
            answer="Revenue answer",
            confidence=0.9,
            latency_ms=80,
            tier1_calls=1,
            tier2_calls=2,
            cache_hit=False,
        ),
        QueryResult(
            query_id="q2",
            original_query=cases[1].query,
            rewritten_query=cases[1].query,
            intent_type="multi_doc",
            traversal_path=["n2"],
            source_nodes=[],
            answer="Comparison answer",
            confidence=0.7,
            latency_ms=120,
            tier1_calls=2,
            tier2_calls=3,
            cache_hit=True,
        ),
        QueryResult(
            query_id="q3",
            original_query=cases[2].query,
            rewritten_query=cases[2].query,
            intent_type="graph",
            traversal_path=["n3"],
            source_nodes=[],
            answer="Graph answer",
            confidence=0.6,
            latency_ms=140,
            tier1_calls=0,
            tier2_calls=0,
            cache_hit=False,
        ),
    ]

    summary = harness.score_results(cases, results)
    assert summary.total_cases == 3
    assert summary.intent_accuracy == 1.0
    assert 0.0 < summary.citation_coverage < 1.0
    assert summary.avg_latency_ms > 0
    assert summary.cache_hit_rate > 0
