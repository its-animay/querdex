from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from querdex.schemas import Section


class URLParser:
    source_format = "url"

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        # `path` can be a plain text file containing a URL or a URL-like string path.
        raw = str(path)
        if path.exists() and path.is_file():
            raw = path.read_text(encoding="utf-8").strip()

        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            msg = f"URL parser expects http/https URL. Got: {raw}"
            raise ValueError(msg)

        req = Request(raw, headers={"User-Agent": "QuerdexBot/0.1"})
        with urlopen(req, timeout=10) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", errors="ignore")

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        sections: list[Section] = []
        page_no = 1
        for block in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = block.get_text(" ", strip=True)
            if not text:
                continue
            links = [a.get("href") for a in block.find_all("a") if a.get("href")]
            sections.append(
                Section(
                    section_id=f"sec_{len(sections) + 1:04d}",
                    doc_id=doc_id,
                    content=text,
                    page_number=page_no,
                    source_format=self.source_format,
                    metadata={
                        "url": raw,
                        "tag": block.name,
                        "links": links,
                    },
                )
            )
            page_no += 1

        if not sections:
            text = soup.get_text(" ", strip=True)
            if text:
                sections.append(
                    Section(
                        section_id="sec_0001",
                        doc_id=doc_id,
                        content=text,
                        page_number=1,
                        source_format=self.source_format,
                        metadata={"url": raw, "tag": "document"},
                    )
                )
        return sections
