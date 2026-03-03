from __future__ import annotations

import asyncio
from pathlib import Path

from querdex.services import build_engine


def _write_markdown(path: Path, liabilities: int) -> None:
    path.write_text(
        f"""# Financial Summary

Q3 revenue was 120 million dollars.

## Liabilities

Total liabilities were {liabilities} million dollars.
""",
        encoding="utf-8",
    )


def test_engine_reindex_updates_version_and_content(tmp_path: Path) -> None:
    db_path = tmp_path / "querdex.db"
    doc_path = tmp_path / "sample.md"
    _write_markdown(doc_path, liabilities=50)

    engine = build_engine(db_path)
    try:
        first = asyncio.run(engine.index_document(doc_path, doc_id="doc_reindex"))
        assert first.version == 1

        _write_markdown(doc_path, liabilities=75)
        second = asyncio.run(engine.reindex_document(doc_path, doc_id="doc_reindex"))
        assert second.version == 2
        assert engine.store.current_version("doc_reindex") == 2

        sections = engine.store.sections_for_doc("doc_reindex")
        assert any("75 million" in section.content for section in sections)
    finally:
        engine.store.close()


def test_engine_reindex_no_changes_is_noop(tmp_path: Path) -> None:
    db_path = tmp_path / "querdex.db"
    doc_path = tmp_path / "sample.md"
    _write_markdown(doc_path, liabilities=50)

    engine = build_engine(db_path)
    try:
        first = asyncio.run(engine.index_document(doc_path, doc_id="doc_reindex_noop"))
        second = asyncio.run(engine.reindex_document(doc_path, doc_id="doc_reindex_noop"))

        assert first.version == second.version == 1
    finally:
        engine.store.close()
