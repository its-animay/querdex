from __future__ import annotations

from pathlib import Path

from querdex.schemas import Section


class TextParser:
    source_format = "text"

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        text = path.read_text(encoding="utf-8")
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        if not chunks:
            chunks = [text.strip()] if text.strip() else []

        sections: list[Section] = []
        for idx, content in enumerate(chunks, start=1):
            sections.append(
                Section(
                    section_id=f"sec_{idx:04d}",
                    doc_id=doc_id,
                    content=content,
                    page_number=idx,
                    source_format=self.source_format,
                    metadata={"chunk_type": "paragraph"},
                )
            )
        return sections
