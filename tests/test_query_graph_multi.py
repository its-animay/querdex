from __future__ import annotations

import asyncio

from querdex.query import GraphWalker, build_virtual_super_tree
from querdex.schemas import TreeNode
from querdex.services import build_engine


def _tree(doc_id: str, node_id: str, title: str, start: int, end: int) -> TreeNode:
    return TreeNode(
        node_id=node_id,
        doc_id=doc_id,
        title=title,
        summary=f"{title} summary",
        start_page=start,
        end_page=end,
        depth=0,
    )


def test_virtual_super_tree_combines_roots() -> None:
    doc_roots = [
        ("doc_a", _tree("doc_a", "root_a", "Doc A Root", 1, 2)),
        ("doc_b", _tree("doc_b", "root_b", "Doc B Root", 1, 3)),
    ]
    super_tree = build_virtual_super_tree(doc_roots)

    assert super_tree.node_id == "synthetic_root"
    assert super_tree.doc_id == "multi_doc"
    assert len(super_tree.children) == 2
    assert any("doc_a" in child.node_id for child in super_tree.children)
    assert any("doc_b" in child.node_id for child in super_tree.children)


def test_graph_walker_uses_bfs_with_filters() -> None:
    observed: dict[str, object] = {}

    def _bfs(*, seed_nodes, max_hops, edge_types):  # noqa: ANN001, ANN202
        observed["seed_nodes"] = seed_nodes
        observed["max_hops"] = max_hops
        observed["edge_types"] = edge_types
        return ["doc_a:n1", "doc_a:n2", "doc_b:n3"]

    walked = GraphWalker().walk(
        seed_node_keys=["doc_a:n1"],
        max_hops=2,
        edge_types={"REFERENCES"},
        bfs_fn=_bfs,
    )

    assert observed["seed_nodes"] == ["doc_a:n1"]
    assert observed["max_hops"] == 2
    assert observed["edge_types"] == {"REFERENCES"}
    assert walked.visited_node_ids == ["n1", "n2", "n3"]


def test_engine_multi_doc_and_graph_fallback(tmp_path) -> None:
    db_path = tmp_path / "querdex.db"
    doc_a = tmp_path / "a.md"
    doc_b = tmp_path / "b.md"
    doc_a.write_text("# Finance\nRevenue increased to 120.\n", encoding="utf-8")
    doc_b.write_text("# Finance\nRevenue decreased to 90.\n", encoding="utf-8")

    engine = build_engine(db_path)
    try:
        asyncio.run(engine.index_document(doc_a, doc_id="doc_a"))
        asyncio.run(engine.index_document(doc_b, doc_id="doc_b"))

        multi = engine.query_document("doc_a", "Compare revenue trend between documents")
        assert multi.intent_type == "multi_doc"
        assert "Cross-document summary" in multi.answer
        assert len({source.doc_id for source in multi.source_nodes}) >= 2

        def _raise_bfs(*, seed_nodes, max_hops, edge_types):  # noqa: ANN001, ANN202
            del seed_nodes, max_hops, edge_types
            raise RuntimeError("graph unavailable")

        engine.store.graph_bfs = _raise_bfs  # type: ignore[method-assign]
        graph = engine.query_document("doc_a", "What cites Revenue?")
        assert graph.intent_type == "single_doc"
        assert graph.answer
    finally:
        engine.store.close()
