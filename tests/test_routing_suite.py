from __future__ import annotations

from querdex.query import QueryAnalyzer, QueryRouter


def test_routing_accuracy_suite() -> None:
    analyzer = QueryAnalyzer()
    router = QueryRouter()

    cases = [
        ("What is the revenue?", "single_doc"),
        ("Compare revenue and liabilities", "multi_doc"),
        ("What cites Section 2.1?", "graph"),
    ]

    for query, expected_route in cases:
        analysis = analyzer.analyze(query)
        assert router.route(analysis) == expected_route


def test_rewrite_uses_previous_turn_for_follow_up() -> None:
    analyzer = QueryAnalyzer()
    turns = [
        {
            "turn_id": 1,
            "original_query": "What is revenue?",
            "rewritten_query": "What is revenue in Q3?",
            "answer": "Revenue is 120.",
        }
    ]
    analysis = analyzer.analyze("What about liabilities?", turns)
    assert "Context from previous turn" in analysis.rewritten_query
