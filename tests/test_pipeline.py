from __future__ import annotations

import asyncio

from querdex.services import build_engine


def test_index_and_query_end_to_end(tmp_path) -> None:
    db_path = tmp_path / "querdex.db"
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(
        """# Financial Summary

Q3 revenue was 120 million dollars.

## Liabilities

Total liabilities were 50 million dollars.
""",
        encoding="utf-8",
    )

    engine = build_engine(db_path)
    try:
        document_index = asyncio.run(engine.index_document(doc_path, doc_id="doc_fin"))
        assert document_index.doc_id == "doc_fin"

        first = engine.query_document("doc_fin", "What is Q3 revenue?", session_id="s1")
        assert first.intent_type == "single_doc"
        assert first.source_nodes
        assert first.latency_ms >= 0

        first_repeat = engine.query_document("doc_fin", "What is Q3 revenue?", session_id="s1")
        assert first_repeat.query_id == first.query_id

        follow_up = engine.query_document("doc_fin", "What about liabilities?", session_id="s1")
        assert "Context from previous turn" in follow_up.rewritten_query
        events = engine.store.recent_feedback_events(limit=5)
        assert events
        assert events[0]["query_id"] == follow_up.query_id
    finally:
        engine.store.close()
