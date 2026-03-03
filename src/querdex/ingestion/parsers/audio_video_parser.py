from __future__ import annotations

from pathlib import Path

from querdex.schemas import Section


class AudioVideoParser:
    source_format = "audio_video"

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        # Strategy:
        # 1) Prefer sidecar transcript file (<file>.txt)
        # 2) Fallback to optional Whisper package if installed
        # 3) Raise if no transcription path is available
        transcript = self._read_sidecar(path)
        if transcript is None:
            transcript = self._transcribe_with_whisper(path)

        if transcript is None or not transcript.strip():
            msg = (
                "No transcript found for audio/video input. "
                "Add a sidecar .txt transcript or install openai-whisper."
            )
            raise ValueError(msg)

        lines = [line.strip() for line in transcript.splitlines() if line.strip()]
        if not lines:
            lines = [transcript.strip()]

        sections: list[Section] = []
        for idx, line in enumerate(lines, start=1):
            sections.append(
                Section(
                    section_id=f"sec_{idx:04d}",
                    doc_id=doc_id,
                    content=line,
                    page_number=idx,
                    source_format=self.source_format,
                    metadata={"type": "transcript_line", "media_file": path.name},
                )
            )
        return sections

    @staticmethod
    def _read_sidecar(path: Path) -> str | None:
        sidecar = path.with_suffix(path.suffix + ".txt")
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8")
        alt = path.with_suffix(".txt")
        if alt.exists():
            return alt.read_text(encoding="utf-8")
        return None

    @staticmethod
    def _transcribe_with_whisper(path: Path) -> str | None:
        try:
            import whisper  # type: ignore[import-not-found]
        except ImportError:
            return None

        model = whisper.load_model("base")
        result = model.transcribe(str(path))
        text = result.get("text")
        return str(text) if text else None
