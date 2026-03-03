from __future__ import annotations

import os
from pathlib import Path

from querdex.ingestion.base import Parser
from querdex.ingestion.parsers.audio_video_parser import AudioVideoParser
from querdex.ingestion.parsers.code_parser import JSCodeParser, PythonCodeParser
from querdex.ingestion.parsers.csv_parser import CSVParser
from querdex.ingestion.parsers.docx_parser import DOCXParser
from querdex.ingestion.parsers.html_parser import HTMLParser
from querdex.ingestion.parsers.markdown_parser import MarkdownParser
from querdex.ingestion.parsers.ocr import CloudOCRProvider, NullOCRProvider, TesseractOCRProvider
from querdex.ingestion.parsers.pdf_parser import PDFParser
from querdex.ingestion.parsers.sqlite_parser import SQLiteParser
from querdex.ingestion.parsers.text_parser import TextParser
from querdex.ingestion.parsers.url_parser import URLParser
from querdex.schemas import Section


class IngestionOrchestrator:
    """Maps file extensions to format parsers and normalizes output into Section[]."""

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {
            ".txt": TextParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".html": HTMLParser(),
            ".htm": HTMLParser(),
            ".docx": DOCXParser(),
            ".pdf": PDFParser(ocr_provider=self._build_ocr_provider()),
            ".py": PythonCodeParser(),
            ".js": JSCodeParser(),
            ".ts": JSCodeParser(),
            ".tsx": JSCodeParser(),
            ".jsx": JSCodeParser(),
            ".csv": CSVParser(),
            ".db": SQLiteParser(),
            ".sqlite": SQLiteParser(),
            ".mp3": AudioVideoParser(),
            ".wav": AudioVideoParser(),
            ".m4a": AudioVideoParser(),
            ".mp4": AudioVideoParser(),
            ".mov": AudioVideoParser(),
            ".url": URLParser(),
        }

    def parse(self, path: str | Path, doc_id: str) -> list[Section]:
        file_path = Path(path)

        if self._looks_like_url(str(path)):
            return URLParser().parse(Path(str(path)), doc_id)

        parser = self._parsers.get(file_path.suffix.lower())
        if parser is None:
            msg = f"No parser registered for extension '{file_path.suffix}'"
            raise ValueError(msg)

        sections = parser.parse(file_path, doc_id)
        if not sections:
            msg = f"Parser returned no content for {file_path}"
            raise ValueError(msg)
        return sections

    def register(self, extension: str, parser: Parser) -> None:
        self._parsers[extension.lower()] = parser

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    @staticmethod
    def _build_ocr_provider() -> NullOCRProvider | TesseractOCRProvider | CloudOCRProvider:
        enabled = os.getenv("QUERDEX_OCR_ENABLED", "false").lower() in {"1", "true", "yes"}
        if enabled:
            provider_kind = os.getenv("QUERDEX_OCR_PROVIDER", "tesseract").lower()
            if provider_kind == "cloud":
                endpoint = os.getenv("QUERDEX_OCR_ENDPOINT", "").strip()
                if endpoint:
                    api_key = os.getenv("QUERDEX_OCR_API_KEY")
                    return CloudOCRProvider(endpoint=endpoint, api_key=api_key)
            cmd = os.getenv("QUERDEX_TESSERACT_CMD", "tesseract")
            return TesseractOCRProvider(tesseract_cmd=cmd)
        return NullOCRProvider()
