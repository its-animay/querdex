from __future__ import annotations

import json
from pathlib import Path

from querdex.ingestion import IngestionOrchestrator


def test_parser_golden_manifest_regression() -> None:
    base = Path("tests/fixtures/golden")
    manifest = json.loads((base / "parser_manifest.json").read_text(encoding="utf-8"))
    orchestrator = IngestionOrchestrator()

    for case in manifest:
        path = base / case["file"]
        sections = orchestrator.parse(path, doc_id=case["doc_id"])

        assert len(sections) >= int(case["min_sections"])
        assert all(section.source_format == case["source_format"] for section in sections)
        combined = "\n".join(section.content for section in sections)
        assert case["contains"] in combined
