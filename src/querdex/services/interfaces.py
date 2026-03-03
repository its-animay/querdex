from __future__ import annotations

from pathlib import Path
from typing import Protocol

from querdex.schemas import QueryResult, Section


class IngestionService(Protocol):
    def parse(self, path: str | Path, doc_id: str) -> list[Section]: ...


class IndexingService(Protocol):
    async def build(self, doc_id: str, title: str, sections: list[Section]) -> object: ...


class QueryService(Protocol):
    def query_document(self, doc_id: str, query: str, session_id: str | None = None) -> QueryResult: ...
