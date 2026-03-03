from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from querdex.schemas import Section


class HTMLParser:
    source_format = "html"

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        sections: list[Section] = []
        section_id = 1
        page_number = 1

        for block in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            content = block.get_text(" ", strip=True)
            if not content:
                continue

            links = [a.get("href") for a in block.find_all("a") if a.get("href")]
            sections.append(
                Section(
                    section_id=f"sec_{section_id:04d}",
                    doc_id=doc_id,
                    content=content,
                    page_number=page_number,
                    source_format=self.source_format,
                    metadata={
                        "tag": block.name,
                        "links": links,
                    },
                )
            )
            section_id += 1
            page_number += 1

        return sections
