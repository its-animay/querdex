from __future__ import annotations

from dataclasses import dataclass

from querdex.storage import SQLiteStore


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    db_connected: bool
    docs_indexed: int


class HealthChecker:
    def check(self, store: SQLiteStore) -> HealthStatus:
        docs = store.all_doc_ids()
        return HealthStatus(ok=True, db_connected=True, docs_indexed=len(docs))
