from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from querdex.schemas import Section


class OCRProvider(Protocol):
    """Hook interface for OCR on scanned PDF pages."""

    def ocr_page(
        self,
        *,
        pdf_path: Path,
        page_number: int,
        page_image_png: bytes,
    ) -> str | None: ...


@dataclass(frozen=True)
class PDFParserConfig:
    include_images: bool = True
    include_tables: bool = True
    max_images_per_page: int = 5
    max_tables_per_page: int = 3


class PDFParser:
    source_format = "pdf"

    def __init__(
        self,
        *,
        ocr_provider: OCRProvider | None = None,
        config: PDFParserConfig | None = None,
    ) -> None:
        self.ocr_provider = ocr_provider
        self.config = config or PDFParserConfig()

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = "PyMuPDF is required for PDF parsing. Install pymupdf to enable PDF support."
            raise RuntimeError(msg) from exc

        sections: list[Section] = []
        with fitz.open(path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                page_images = self._extract_page_images(pdf, page)
                table_contents = self._extract_tables(page, text)

                if not text and self.ocr_provider is not None:
                    page_image = page.get_pixmap().tobytes("png")
                    ocr_text = self.ocr_provider.ocr_page(
                        pdf_path=path,
                        page_number=page_index,
                        page_image_png=page_image,
                    )
                    if ocr_text:
                        text = ocr_text.strip()

                if text:
                    sections.append(
                        Section(
                            section_id=f"sec_{len(sections) + 1:04d}",
                            doc_id=doc_id,
                            content=text,
                            page_number=page_index,
                            source_format=self.source_format,
                            metadata={
                                "page": page_index,
                                "type": "text",
                                "image_count": len(page_images),
                                "table_count": len(table_contents),
                            },
                        )
                    )

                if self.config.include_tables:
                    for table_index, table_text in enumerate(
                        table_contents[: self.config.max_tables_per_page], start=1
                    ):
                        sections.append(
                            Section(
                                section_id=f"sec_{len(sections) + 1:04d}",
                                doc_id=doc_id,
                                content=table_text,
                                page_number=page_index,
                                source_format=self.source_format,
                                metadata={
                                    "page": page_index,
                                    "type": "table",
                                    "table_index": table_index,
                                },
                            )
                        )

                if self.config.include_images:
                    for image_index, image_bytes in enumerate(page_images[: self.config.max_images_per_page], start=1):
                        sections.append(
                            Section(
                                section_id=f"sec_{len(sections) + 1:04d}",
                                doc_id=doc_id,
                                content=f"Image {image_index} from page {page_index}",
                                page_number=page_index,
                                source_format=self.source_format,
                                raw_bytes=image_bytes,
                                metadata={
                                    "page": page_index,
                                    "type": "image",
                                    "image_index": image_index,
                                    "mime_type": "image/png",
                                },
                            )
                        )

        if not sections:
            msg = f"No extractable sections found in PDF {path}"
            raise ValueError(msg)
        return sections

    def _extract_page_images(self, pdf: Any, page: Any) -> list[bytes]:
        images: list[bytes] = []
        for image_ref in page.get_images(full=True):
            xref = int(image_ref[0])
            image_data = pdf.extract_image(xref)
            data = image_data.get("image")
            if isinstance(data, bytes) and data:
                images.append(data)
        return images

    def _extract_tables(self, page: Any, page_text: str) -> list[str]:
        # Prefer native table detection when available in PyMuPDF versions that expose it.
        tables = self._extract_tables_with_fitz(page)
        if tables:
            return tables
        # Fallback heuristic for environments without native table extraction.
        return self._extract_tables_with_heuristic(page_text)

    def _extract_tables_with_fitz(self, page: Any) -> list[str]:
        if not hasattr(page, "find_tables"):
            return []
        try:
            finder = page.find_tables()
        except Exception:
            return []

        extracted: list[str] = []
        for table in finder.tables:
            try:
                matrix = table.extract()
            except Exception:
                continue
            if not matrix:
                continue
            rows = []
            for row in matrix:
                cells = [str(cell or "").strip() for cell in row]
                rows.append(" | ".join(cells))
            table_text = "\n".join(row for row in rows if row.strip()).strip()
            if table_text:
                extracted.append(table_text)
        return extracted

    @staticmethod
    def _extract_tables_with_heuristic(page_text: str) -> list[str]:
        candidates: list[str] = []
        lines = [line.rstrip() for line in page_text.splitlines()]
        active: list[str] = []

        for line in lines:
            # Heuristic: multiple spaces with numeric-heavy cells resemble tabular rows.
            looks_tabular = ("  " in line) and any(ch.isdigit() for ch in line)
            if looks_tabular:
                active.append(" | ".join(part.strip() for part in line.split("  ") if part.strip()))
            else:
                if len(active) >= 2:
                    candidates.append("\n".join(active))
                active = []

        if len(active) >= 2:
            candidates.append("\n".join(active))

        return candidates
