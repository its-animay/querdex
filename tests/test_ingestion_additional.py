from __future__ import annotations

import sqlite3
from pathlib import Path

from querdex.ingestion import IngestionOrchestrator


def test_ingestion_code_parsers_python_and_js(tmp_path: Path) -> None:
    py_path = tmp_path / "sample.py"
    py_path.write_text(
        """
class RevenueModel:
    def forecast(self):
        return 120

def liabilities():
    return 50
""".strip(),
        encoding="utf-8",
    )
    js_path = tmp_path / "sample.js"
    js_path.write_text(
        """
function getRevenue() { return 120; }
class DebtTracker {}
""".strip(),
        encoding="utf-8",
    )

    orchestrator = IngestionOrchestrator()
    py_sections = orchestrator.parse(py_path, doc_id="doc_code")
    js_sections = orchestrator.parse(js_path, doc_id="doc_code")

    assert py_sections
    assert any(section.metadata.get("kind") == "class" for section in py_sections)
    assert any(section.metadata.get("kind") == "function" for section in py_sections)
    assert js_sections
    assert any(section.metadata.get("name") == "getRevenue" for section in js_sections)


def test_ingestion_csv_parser_includes_schema_and_stats(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "quarter,revenue,liabilities\nQ1,100,45\nQ2,110,47\nQ3,120,50\n",
        encoding="utf-8",
    )

    orchestrator = IngestionOrchestrator()
    sections = orchestrator.parse(csv_path, doc_id="doc_csv")

    assert sections[0].metadata.get("type") == "schema"
    assert "quarter" in sections[0].content
    assert any(section.metadata.get("type") == "row" for section in sections)
    assert any(section.metadata.get("type") == "stats" for section in sections)


def test_ingestion_audio_parser_uses_sidecar_transcript(tmp_path: Path) -> None:
    media_path = tmp_path / "meeting.mp3"
    media_path.write_bytes(b"fake-audio")
    sidecar = tmp_path / "meeting.mp3.txt"
    sidecar.write_text("Line one\nLine two", encoding="utf-8")

    orchestrator = IngestionOrchestrator()
    sections = orchestrator.parse(media_path, doc_id="doc_media")

    assert len(sections) == 2
    assert sections[0].source_format == "audio_video"
    assert sections[0].metadata.get("media_file") == "meeting.mp3"


def test_ingestion_url_parser_fetches_and_normalizes(monkeypatch, tmp_path: Path) -> None:
    html = "<html><body><h1>Title</h1><p>Revenue is 120.</p></body></html>"

    def _fake_fetch_html(url, timeout=10.0):  # noqa: ANN001, ANN202
        del url, timeout
        return html

    monkeypatch.setattr("querdex.ingestion.parsers.url_parser._fetch_html", _fake_fetch_html)

    url_file = tmp_path / "target.url"
    url_file.write_text("https://example.com/report", encoding="utf-8")

    orchestrator = IngestionOrchestrator()
    sections = orchestrator.parse(url_file, doc_id="doc_url")

    assert sections
    assert sections[0].source_format == "url"
    assert any("Revenue is 120" in section.content for section in sections)


def test_ingestion_sqlite_parser_reads_schema_and_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE metrics (quarter TEXT, revenue INTEGER)")
        conn.execute("INSERT INTO metrics(quarter, revenue) VALUES ('Q1', 100), ('Q2', 110)")
        conn.commit()
    finally:
        conn.close()

    orchestrator = IngestionOrchestrator()
    sections = orchestrator.parse(db_path, doc_id="doc_sqlite")

    assert sections
    assert sections[0].source_format == "sqlite"
    assert "Table metrics columns" in sections[0].content
    assert any("quarter=Q1" in section.content for section in sections)
