from __future__ import annotations

import sqlite3
from pathlib import Path

from querdex.schemas import Section


class SQLiteParser:
    source_format = "sqlite"

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            table_names = [str(row["name"]) for row in tables]
            if not table_names:
                msg = f"No user tables found in SQLite database: {path}"
                raise ValueError(msg)

            sections: list[Section] = []
            page = 1
            for table in table_names:
                info_rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
                columns = [str(row["name"]) for row in info_rows]
                sections.append(
                    Section(
                        section_id=f"sec_{len(sections) + 1:04d}",
                        doc_id=doc_id,
                        content=f"Table {table} columns: {', '.join(columns) if columns else '(none)'}",
                        page_number=page,
                        source_format=self.source_format,
                        metadata={"type": "table_schema", "table": table, "column_count": len(columns)},
                    )
                )
                page += 1

                preview = conn.execute(f"SELECT * FROM '{table}' LIMIT 25").fetchall()
                for idx, row in enumerate(preview, start=1):
                    text = "; ".join(f"{key}={row[key]}" for key in row.keys())
                    sections.append(
                        Section(
                            section_id=f"sec_{len(sections) + 1:04d}",
                            doc_id=doc_id,
                            content=text,
                            page_number=page,
                            source_format=self.source_format,
                            metadata={"type": "table_row", "table": table, "row_index": idx},
                        )
                    )
                    page += 1

            return sections
        finally:
            conn.close()
