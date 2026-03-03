from __future__ import annotations

from pathlib import Path
from typing import Protocol

from querdex.schemas import Section


class Parser(Protocol):
    source_format: str

    def parse(self, path: Path, doc_id: str) -> list[Section]: ...
