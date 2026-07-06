from __future__ import annotations

import re
from dataclasses import dataclass, field

from querdex.schemas import Section

_SENTENCE_END_RE = re.compile(r"[.!?]\s")
_PIECE_SEPARATOR = "\n\n"


@dataclass
class ChunkPiece:
    """A contiguous region of a chunk mapped back to its source section."""

    section_id: str
    page_number: int
    chunk_offset: int
    section_offset: int
    length: int


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    pieces: list[ChunkPiece] = field(default_factory=list)

    def locate(self, char_start: int) -> ChunkPiece | None:
        """Return the piece containing char_start (chunk coordinates)."""
        for piece in self.pieces:
            if piece.chunk_offset <= char_start < piece.chunk_offset + piece.length:
                return piece
        return None


def _split_long_text(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split text into (section_offset, piece) tuples of at most max_chars.

    Cuts prefer sentence boundaries, then word boundaries, so extractions
    are unlikely to straddle a chunk edge.
    """
    pieces: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_chars:
            pieces.append((start, text[start:]))
            break
        window = text[start : start + max_chars]
        cut = max((m.end() for m in _SENTENCE_END_RE.finditer(window)), default=0)
        if cut < max_chars // 2:
            space = window.rfind(" ")
            cut = space + 1 if space > max_chars // 2 else max_chars
        pieces.append((start, text[start : start + cut]))
        start += cut
    return pieces


def chunk_sections(sections: list[Section], *, max_chars: int = 4000) -> list[TextChunk]:
    """Group section contents into chunks of at most max_chars.

    Consecutive small sections are packed together; oversized sections are
    split at sentence boundaries. Every region of every chunk maps back to
    its source section and offset so extractions can be grounded precisely.
    """
    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_pieces: list[ChunkPiece] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_parts, current_pieces, current_len
        if current_parts:
            chunks.append(
                TextChunk(
                    chunk_id=f"chunk_{len(chunks) + 1:04d}",
                    text="".join(current_parts),
                    pieces=current_pieces,
                )
            )
        current_parts = []
        current_pieces = []
        current_len = 0

    def add_piece(section: Section, section_offset: int, piece_text: str) -> None:
        nonlocal current_len
        if current_parts:
            current_parts.append(_PIECE_SEPARATOR)
            current_len += len(_PIECE_SEPARATOR)
        current_pieces.append(
            ChunkPiece(
                section_id=section.section_id,
                page_number=section.page_number,
                chunk_offset=current_len,
                section_offset=section_offset,
                length=len(piece_text),
            )
        )
        current_parts.append(piece_text)
        current_len += len(piece_text)

    for section in sections:
        for section_offset, piece_text in _split_long_text(section.content, max_chars):
            if current_len and current_len + len(_PIECE_SEPARATOR) + len(piece_text) > max_chars:
                flush()
            add_piece(section, section_offset, piece_text)
    flush()
    return chunks
