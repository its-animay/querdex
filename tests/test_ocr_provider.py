from __future__ import annotations

from pathlib import Path

from querdex.ingestion.orchestrator import IngestionOrchestrator
from querdex.ingestion.parsers.ocr import CloudOCRProvider, NullOCRProvider, TesseractOCRProvider


def test_orchestrator_selects_null_ocr_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("QUERDEX_OCR_ENABLED", "false")
    provider = IngestionOrchestrator._build_ocr_provider()
    assert isinstance(provider, NullOCRProvider)


def test_orchestrator_selects_tesseract_by_default(monkeypatch) -> None:
    monkeypatch.setenv("QUERDEX_OCR_ENABLED", "true")
    monkeypatch.delenv("QUERDEX_OCR_PROVIDER", raising=False)
    monkeypatch.setenv("QUERDEX_TESSERACT_CMD", "tesseract-custom")
    provider = IngestionOrchestrator._build_ocr_provider()
    assert isinstance(provider, TesseractOCRProvider)
    assert provider.tesseract_cmd == "tesseract-custom"


def test_orchestrator_selects_cloud_ocr_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("QUERDEX_OCR_ENABLED", "true")
    monkeypatch.setenv("QUERDEX_OCR_PROVIDER", "cloud")
    monkeypatch.setenv("QUERDEX_OCR_ENDPOINT", "https://ocr.example.com/v1/ocr")
    monkeypatch.setenv("QUERDEX_OCR_API_KEY", "key123")
    provider = IngestionOrchestrator._build_ocr_provider()
    assert isinstance(provider, CloudOCRProvider)
    assert provider.endpoint.endswith("/ocr")


def test_cloud_ocr_provider_parses_response(monkeypatch) -> None:
    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return b'{"text":"OCR text"}'

    def _fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        del timeout
        assert req.full_url == "https://ocr.example.com/v1/ocr"
        return _Resp()

    monkeypatch.setattr("querdex.ingestion.parsers.ocr.urlopen", _fake_urlopen)
    provider = CloudOCRProvider(endpoint="https://ocr.example.com/v1/ocr", api_key="key123")
    text = provider.ocr_page(pdf_path=Path("x.pdf"), page_number=1, page_image_png=b"img")
    assert text == "OCR text"
