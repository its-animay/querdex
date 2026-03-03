from __future__ import annotations

from querdex.indexing import IndexBuilderCoordinator
from querdex.schemas import QueryResult, Section
from querdex.storage import SQLiteStore


def test_section_store_and_query_result_persistence(tmp_path) -> None:
    db_path = tmp_path / "querdex.db"
    store = SQLiteStore(db_path)

    sections = [
        Section(
            section_id="sec_0001",
            doc_id="doc_a",
            content="Revenue increased to 120 in Q3.",
            page_number=1,
            source_format="text",
            metadata={},
        ),
        Section(
            section_id="sec_0002",
            doc_id="doc_a",
            content="Liabilities were reduced.",
            page_number=2,
            source_format="text",
            metadata={},
        ),
    ]

    coordinator = IndexBuilderCoordinator()
    import asyncio

    indexing = asyncio.run(coordinator.build("doc_a", "Doc A", sections))
    store.save_index(
        doc_id="doc_a",
        title="Doc A",
        source_format="text",
        sections=sections,
        indexing_result=indexing,
    )

    fetched = store.fetch_sections_by_page_range("doc_a", 1, 2)
    assert len(fetched) == 2
    assert fetched[0].content.startswith("Revenue")

    result = QueryResult(
        query_id="q_1",
        original_query="What is revenue?",
        rewritten_query="What is revenue?",
        intent_type="single_doc",
        traversal_path=["node_0001"],
        answer="Revenue increased.",
        confidence=0.8,
        latency_ms=0,
        tier1_calls=1,
        tier2_calls=1,
        cache_hit=False,
    )
    store.save_query_result(result, session_id="sess_1")
    recent = store.recent_query_results(limit=1)
    assert len(recent) == 1
    assert recent[0].query_id == "q_1"

    store.add_session_turn("sess_1", "orig", "rewritten", "answer")
    turns = store.recent_session_turns("sess_1")
    assert turns
    assert turns[0]["rewritten_query"] == "rewritten"

    store.close()
