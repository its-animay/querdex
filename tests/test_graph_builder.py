from __future__ import annotations

from querdex.indexing import KnowledgeGraphBuilder
from querdex.schemas import EntityRef, TreeNode


def _tree() -> TreeNode:
    return TreeNode(
        node_id="root",
        doc_id="doc_a",
        title="Root",
        summary="overview",
        start_page=1,
        end_page=2,
        depth=0,
        children=[
            TreeNode(
                node_id="n1",
                doc_id="doc_a",
                title="Revenue 2023 growth",
                summary="Profit increase in 2023",
                start_page=1,
                end_page=1,
                depth=1,
            ),
            TreeNode(
                node_id="n2",
                doc_id="doc_a",
                title="Revenue 2024 decline",
                summary="Profit drop in 2024",
                start_page=2,
                end_page=2,
                depth=1,
            ),
        ],
    )


def test_graph_builder_creates_core_edge_types() -> None:
    builder = KnowledgeGraphBuilder()
    root = _tree()
    nodes, edges = builder.build(
        root,
        entity_map={
            "Section 1": [EntityRef(doc_id="doc_a", node_id="n2")],
        },
        cross_refs={"n1": ["Section 1"]},
    )

    edge_types = {edge_type for _source, _target, edge_type, _weight in edges}
    assert ("doc_a", "root") in nodes
    assert ("doc_a", "n1") in nodes
    assert "ELABORATES" in edge_types
    assert "REFERENCES" in edge_types
    assert "TEMPORALLY_FOLLOWS" in edge_types
    assert "CONTRADICTS" in edge_types


def test_cross_doc_similarity_edges_use_embedding_similarity() -> None:
    builder = KnowledgeGraphBuilder()
    edges = builder.cross_doc_similarity_edges(
        source_doc_id="doc_a",
        source_nodes=[("doc_a", "a1", "Revenue growth and operating margin")],
        other_docs_nodes=[
            ("doc_b", "b1", "Operating margin and revenue growth trends"),
            ("doc_c", "c1", "Unrelated botanical classification text"),
        ],
        threshold=0.3,
    )

    keys = {(source, target, edge_type) for source, target, edge_type, _weight in edges}
    assert ("doc_a:a1", "doc_b:b1", "CROSS_DOC_SIMILAR") in keys
    assert ("doc_a:a1", "doc_c:c1", "CROSS_DOC_SIMILAR") not in keys
