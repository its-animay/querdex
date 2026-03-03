from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from querdex.ingestion import IngestionOrchestrator
from querdex.services import build_engine

fitz = pytest.importorskip("fitz")


def _create_real_pdf(path: Path) -> None:
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Tiered Search uses Tier 1 fast prune and Tier 2 deep reasoning for retrieval.",
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Query cache stores cluster-aware scores and avoids repeated expensive calls.",
    )
    doc.save(str(path))
    doc.close()


def test_ingestion_orchestrator_parses_real_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "integration.pdf"
    _create_real_pdf(pdf_path)

    orchestrator = IngestionOrchestrator()
    sections = orchestrator.parse(pdf_path, doc_id="doc_pdf_integration")

    assert sections
    assert any(section.source_format == "pdf" for section in sections)
    assert any("Tiered Search" in section.content for section in sections)


def test_engine_index_and_query_real_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "integration.pdf"
    _create_real_pdf(pdf_path)
    db_path = tmp_path / "querdex.db"

    engine = build_engine(db_path)
    try:
        document_index = asyncio.run(engine.index_document(pdf_path, doc_id="doc_pdf_integration"))
        assert document_index.doc_id == "doc_pdf_integration"

        result = engine.query_document(
            "doc_pdf_integration",
            "How does tiered search retrieval work and what does cache do?",
            session_id="pdf_session",
        )

        assert result.source_nodes
        assert "No relevant content found" not in result.answer
        assert result.intent_type == "single_doc"
    finally:
        engine.store.close()
