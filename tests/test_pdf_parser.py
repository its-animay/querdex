from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from querdex.ingestion.parsers.pdf_parser import OCRProvider, PDFParser, PDFParserConfig


@dataclass
class FakePixmap:
    data: bytes = b"fake-png-bytes"

    def tobytes(self, fmt: str) -> bytes:
        assert fmt == "png"
        return self.data


class FakeTable:
    def __init__(self, matrix: list[list[str]]) -> None:
        self._matrix = matrix

    def extract(self) -> list[list[str]]:
        return self._matrix


class FakeFinder:
    def __init__(self, matrices: list[list[list[str]]]) -> None:
        self.tables = [FakeTable(matrix) for matrix in matrices]


class FakePageBase:
    def __init__(self, *, text: str, image_refs: list[int] | None = None, pixmap: bytes = b"fake-png-bytes") -> None:
        self._text = text
        self._image_refs = image_refs or []
        self._pixmap = pixmap

    def get_text(self, mode: str) -> str:
        assert mode == "text"
        return self._text

    def get_images(self, full: bool = True) -> list[tuple[int, int, int, int, int, int, int, int, int, int]]:
        assert full
        return [(xref, 0, 0, 0, 0, 0, 0, 0, 0, 0) for xref in self._image_refs]

    def get_pixmap(self) -> FakePixmap:
        return FakePixmap(self._pixmap)


class FakePageWithTables(FakePageBase):
    def __init__(
        self, *, text: str, table_matrices: list[list[list[str]]], image_refs: list[int] | None = None
    ) -> None:
        super().__init__(text=text, image_refs=image_refs)
        self._table_matrices = table_matrices

    def find_tables(self) -> FakeFinder:
        return FakeFinder(self._table_matrices)


class FakeDoc:
    def __init__(
        self,
        path: Path,
        pages: list[Any],
        image_map: dict[int, bytes] | None = None,
    ) -> None:
        self.path = path
        self._pages = pages
        self._image_map = image_map or {}

    def __enter__(self) -> FakeDoc:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __iter__(self):
        return iter(self._pages)

    def extract_image(self, xref: int) -> dict[str, bytes]:
        data = self._image_map.get(xref, b"")
        return {"image": data}


class RecordingOCR(OCRProvider):
    def __init__(self, text: str = "OCR extracted text") -> None:
        self.text = text
        self.calls: list[tuple[Path, int, bytes]] = []

    def ocr_page(self, *, pdf_path: Path, page_number: int, page_image_png: bytes) -> str | None:
        self.calls.append((pdf_path, page_number, page_image_png))
        return self.text


def _install_fake_fitz(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pages: list[Any],
    image_map: dict[int, bytes] | None = None,
) -> None:
    fake_module = SimpleNamespace(open=lambda path: FakeDoc(Path(path), pages, image_map))
    monkeypatch.setitem(sys.modules, "fitz", fake_module)


def test_pdf_parser_extracts_text_section(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_fitz(monkeypatch, pages=[FakePageBase(text="Tiered Search overview")])
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    parser = PDFParser(config=PDFParserConfig(include_images=False, include_tables=False))
    sections = parser.parse(pdf_path, doc_id="doc_pdf")

    assert len(sections) == 1
    assert sections[0].metadata["type"] == "text"
    assert sections[0].content == "Tiered Search overview"


def test_pdf_parser_uses_ocr_when_text_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_fitz(monkeypatch, pages=[FakePageBase(text="")])
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    ocr = RecordingOCR(text="OCR fallback content")
    parser = PDFParser(ocr_provider=ocr, config=PDFParserConfig(include_images=False, include_tables=False))
    sections = parser.parse(pdf_path, doc_id="doc_pdf")

    assert len(sections) == 1
    assert sections[0].content == "OCR fallback content"
    assert len(ocr.calls) == 1
    assert ocr.calls[0][1] == 1


def test_pdf_parser_extracts_tables_from_native_detection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    page = FakePageWithTables(
        text="Financial page",
        table_matrices=[[["Metric", "Value"], ["Revenue", "120"]]],
    )
    _install_fake_fitz(monkeypatch, pages=[page])
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    parser = PDFParser(config=PDFParserConfig(include_images=False, include_tables=True))
    sections = parser.parse(pdf_path, doc_id="doc_pdf")

    table_sections = [section for section in sections if section.metadata.get("type") == "table"]
    assert table_sections
    assert "Metric | Value" in table_sections[0].content


def test_pdf_parser_uses_heuristic_table_detection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    table_like_text = "Metric  2024\nRevenue  120\nCost  80\nNarrative sentence"
    _install_fake_fitz(monkeypatch, pages=[FakePageBase(text=table_like_text)])
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    parser = PDFParser(config=PDFParserConfig(include_images=False, include_tables=True))
    sections = parser.parse(pdf_path, doc_id="doc_pdf")

    table_sections = [section for section in sections if section.metadata.get("type") == "table"]
    assert table_sections
    assert "Revenue | 120" in table_sections[0].content


def test_pdf_parser_extracts_images_and_respects_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    page = FakePageBase(text="Image-heavy page", image_refs=[10, 11])
    _install_fake_fitz(monkeypatch, pages=[page], image_map={10: b"img-a", 11: b"img-b"})
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    parser = PDFParser(config=PDFParserConfig(include_images=True, include_tables=False, max_images_per_page=1))
    sections = parser.parse(pdf_path, doc_id="doc_pdf")

    image_sections = [section for section in sections if section.metadata.get("type") == "image"]
    assert len(image_sections) == 1
    assert image_sections[0].raw_bytes == b"img-a"


def test_pdf_parser_raises_for_empty_pdf_without_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_fitz(monkeypatch, pages=[FakePageBase(text="")], image_map={})
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    parser = PDFParser(config=PDFParserConfig(include_images=False, include_tables=False))
    with pytest.raises(ValueError):
        parser.parse(pdf_path, doc_id="doc_pdf")
