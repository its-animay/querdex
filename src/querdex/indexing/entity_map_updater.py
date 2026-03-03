from __future__ import annotations

from collections import defaultdict

from querdex.schemas import EntityRef


class EntityMapUpdater:
    """Append/merge utility for entity maps with deterministic deduplication."""

    def merge(
        self,
        existing: dict[str, list[EntityRef]],
        incoming: dict[str, list[EntityRef]],
    ) -> dict[str, list[EntityRef]]:
        merged: dict[str, list[EntityRef]] = defaultdict(list)

        for entity, refs in existing.items():
            merged[entity].extend(refs)
        for entity, refs in incoming.items():
            merged[entity].extend(refs)

        return {entity: self._dedupe_refs(refs) for entity, refs in merged.items()}

    def to_rows(self, entity_map: dict[str, list[EntityRef]], *, version: int) -> list[tuple[str, str, str, int]]:
        rows: list[tuple[str, str, str, int]] = []
        for entity, refs in entity_map.items():
            for ref in self._dedupe_refs(refs):
                rows.append((entity, ref.doc_id, ref.node_id, version))
        return rows

    @staticmethod
    def _dedupe_refs(refs: list[EntityRef]) -> list[EntityRef]:
        seen: set[tuple[str, str]] = set()
        deduped: list[EntityRef] = []
        for ref in refs:
            key = (ref.doc_id, ref.node_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped
