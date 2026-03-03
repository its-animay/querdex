from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from querdex.ingestion.parsers.pdf_parser import OCRProvider


class TesseractOCRProvider(OCRProvider):
    """Local OCR provider using system tesseract binary."""

    def __init__(self, tesseract_cmd: str = "tesseract") -> None:
        self.tesseract_cmd = tesseract_cmd

    def ocr_page(self, *, pdf_path: Path, page_number: int, page_image_png: bytes) -> str | None:
        del pdf_path, page_number
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "page.png"
            out_base = Path(tmpdir) / "ocr_out"
            image_path.write_bytes(page_image_png)
            cmd = [self.tesseract_cmd, str(image_path), str(out_base), "-l", "eng"]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return None
            txt_path = out_base.with_suffix(".txt")
            if not txt_path.exists():
                return None
            text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            return text or None


class NullOCRProvider(OCRProvider):
    def ocr_page(self, *, pdf_path: Path, page_number: int, page_image_png: bytes) -> str | None:
        del pdf_path, page_number, page_image_png
        return None


class CloudOCRProvider(OCRProvider):
    """HTTP-based OCR adapter for external OCR services."""

    def __init__(self, endpoint: str, api_key: str | None = None, timeout_seconds: float = 20.0) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def ocr_page(self, *, pdf_path: Path, page_number: int, page_image_png: bytes) -> str | None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "filename": pdf_path.name,
            "page_number": page_number,
            "image_base64": base64.b64encode(page_image_png).decode("ascii"),
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                body = response.read().decode("utf-8", errors="ignore")
        except Exception:
            return None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        text = parsed.get("text")
        return str(text).strip() if text else None
