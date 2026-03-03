from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

from querdex.schemas import Section


class MarkdownParser:
    source_format = "markdown"

    def __init__(self) -> None:
        self._md = MarkdownIt()

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        text = path.read_text(encoding="utf-8")
        tokens = self._md.parse(text)

        sections: list[Section] = []
        current_heading = "Introduction"
        buffer: list[str] = []
        page_number = 1

        def flush() -> None:
            nonlocal page_number
            content = "\n".join(buffer).strip()
            if not content:
                return
            section_id = f"sec_{len(sections) + 1:04d}"
            sections.append(
                Section(
                    section_id=section_id,
                    doc_id=doc_id,
                    content=content,
                    page_number=page_number,
                    source_format=self.source_format,
                    metadata={"heading": current_heading},
                )
            )
            page_number += 1

        for token in tokens:
            if token.type == "heading_open":
                flush()
                buffer = []
            elif token.type == "inline" and token.map is not None:
                if token.level == 1:
                    current_heading = token.content
                else:
                    buffer.append(token.content)

        flush()
        if not sections and text.strip():
            sections.append(
                Section(
                    section_id="sec_0001",
                    doc_id=doc_id,
                    content=text.strip(),
                    page_number=1,
                    source_format=self.source_format,
                    metadata={"heading": current_heading},
                )
            )
        return sections
