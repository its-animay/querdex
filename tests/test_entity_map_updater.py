from __future__ import annotations

from querdex.indexing import EntityMapUpdater
from querdex.schemas import EntityRef


def test_entity_map_updater_merges_and_dedupes() -> None:
    updater = EntityMapUpdater()

    existing = {
        "Tesla": [EntityRef(doc_id="doc_a", node_id="node_0001")],
    }
    incoming = {
        "Tesla": [
            EntityRef(doc_id="doc_a", node_id="node_0001"),
            EntityRef(doc_id="doc_b", node_id="node_0003"),
        ],
        "Revenue": [EntityRef(doc_id="doc_b", node_id="node_0004")],
    }

    merged = updater.merge(existing, incoming)

    assert "Tesla" in merged
    assert len(merged["Tesla"]) == 2
    assert {ref.doc_id for ref in merged["Tesla"]} == {"doc_a", "doc_b"}
    assert "Revenue" in merged
