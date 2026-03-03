from __future__ import annotations

import csv
from pathlib import Path

from querdex.schemas import Section


class CSVParser:
    source_format = "csv"

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        sections: list[Section] = []
        schema_desc = ", ".join(fieldnames) if fieldnames else "No columns"
        sections.append(
            Section(
                section_id="sec_0001",
                doc_id=doc_id,
                content=f"CSV schema columns: {schema_desc}",
                page_number=1,
                source_format=self.source_format,
                metadata={"type": "schema", "column_count": len(fieldnames)},
            )
        )

        preview_rows = rows[:25]
        for idx, row in enumerate(preview_rows, start=2):
            entries = "; ".join(f"{k}={v}" for k, v in row.items())
            sections.append(
                Section(
                    section_id=f"sec_{len(sections) + 1:04d}",
                    doc_id=doc_id,
                    content=entries,
                    page_number=idx,
                    source_format=self.source_format,
                    metadata={"type": "row", "row_index": idx - 1},
                )
            )

        stats = self._build_column_stats(rows, fieldnames)
        if stats:
            sections.append(
                Section(
                    section_id=f"sec_{len(sections) + 1:04d}",
                    doc_id=doc_id,
                    content=stats,
                    page_number=len(sections) + 1,
                    source_format=self.source_format,
                    metadata={"type": "stats"},
                )
            )

        return sections

    @staticmethod
    def _build_column_stats(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
        if not rows or not fieldnames:
            return ""

        chunks: list[str] = []
        for name in fieldnames:
            values = [row.get(name, "") for row in rows]
            non_empty = sum(1 for v in values if str(v).strip())
            unique = len({str(v).strip() for v in values if str(v).strip()})
            chunks.append(f"{name}: non_empty={non_empty}, unique={unique}")
        return "Column stats -> " + " | ".join(chunks)
