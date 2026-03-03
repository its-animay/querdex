from __future__ import annotations

from querdex.storage import NetworkXGraphStore


def test_graph_store_persists_and_filters_edges(tmp_path) -> None:
    path = tmp_path / "graph.pkl"
    store = NetworkXGraphStore(path)
    store.upsert_document_graph(
        doc_id="doc_a",
        nodes=[("doc_a", "n1"), ("doc_a", "n2"), ("doc_b", "n3")],
        edges=[
            ("n1", "n2", "ELABORATES", 0.9),
            ("doc_a:n2", "doc_b:n3", "CROSS_DOC_SIMILAR", 0.4),
        ],
    )

    reloaded = NetworkXGraphStore(path)
    assert "doc_a:n2" in reloaded.neighbors("doc_a:n1")
    assert reloaded.neighbors("doc_a:n2", edge_types={"CROSS_DOC_SIMILAR"}) == ["doc_b:n3"]
    assert reloaded.bfs(seed_nodes=["doc_a:n1"], max_hops=2)
