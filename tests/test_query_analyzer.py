from __future__ import annotations

from querdex.query import QueryAnalyzer


def test_query_analyzer_ignores_interrogative_stopwords_as_entities() -> None:
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze("What is the retrieval strategy?")

    assert "What" not in analysis.extracted_entities
