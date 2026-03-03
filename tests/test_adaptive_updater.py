from __future__ import annotations

from querdex.adaptive import AdaptiveUpdater
from querdex.schemas import IndexingResult, QueryResult, Section, SourceNode, TreeNode
from querdex.storage import SQLiteStore
from querdex.utils import find_node


def _base_tree() -> TreeNode:
    child = TreeNode(
        node_id="node_child",
        doc_id="doc_adapt",
        title="Liabilities",
        summary="Old summary",
        start_page=2,
        end_page=2,
        depth=1,
    )
    return TreeNode(
        node_id="node_root",
        doc_id="doc_adapt",
        title="Finance",
        summary="Finance root",
        start_page=1,
        end_page=2,
        depth=0,
        children=[child],
    )


def test_adaptive_updater_updates_metrics_affinity_and_summaries(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "querdex.db")
    try:
        sections = [
            Section(
                section_id="sec_0001",
                doc_id="doc_adapt",
                content="Revenue is strong.",
                page_number=1,
                source_format="text",
                metadata={},
            ),
            Section(
                section_id="sec_0002",
                doc_id="doc_adapt",
                content="Liabilities are stable.",
                page_number=2,
                source_format="text",
                metadata={},
            ),
        ]
        tree = _base_tree()
        store.save_index(
            doc_id="doc_adapt",
            title="Adaptive",
            source_format="text",
            sections=sections,
            indexing_result=IndexingResult(
                tree_root=tree,
                entity_map={},
                graph_nodes=[("doc_adapt", "node_root"), ("doc_adapt", "node_child")],
                graph_edges=[("node_root", "node_child", "ELABORATES", 0.9)],
            ),
        )

        query_results = [
            QueryResult(
                query_id=f"q{i}",
                original_query="Liabilities?",
                rewritten_query="Liabilities details",
                intent_type="single_doc",
                traversal_path=["node_root", "node_child"],
                source_nodes=[
                    SourceNode(
                        node_id="node_child",
                        doc_id="doc_adapt",
                        title="Liabilities",
                        pages="2-2",
                        confidence=0.8,
                    )
                ],
                answer="Answer",
                confidence=0.8,
                latency_ms=10,
                tier1_calls=1,
                tier2_calls=1,
                cache_hit=False,
            )
            for i in range(3)
        ]
        feedback_events = [
            {
                "query_id": f"q{i}",
                "doc_id": "doc_adapt",
                "visited_nodes": ["node_child"],
                "used_nodes": [],
            }
            for i in range(10)
        ]

        updated_tree, misleading = AdaptiveUpdater().update_tree(
            doc_id="doc_adapt",
            tree_root=tree,
            query_results=query_results,
            feedback_events=feedback_events,
            store=store,
        )

        assert any(item.node_id == "node_child" for item in misleading)
        metrics = store.node_metrics_for_doc("doc_adapt")
        assert any(item["node_id"] == "node_child" and item["visit_count"] >= 10 for item in metrics)

        affinity = store.affinity_scores_for_doc("doc_adapt")
        assert "node_child" in affinity
        assert affinity["node_child"]

        updated_child = find_node(updated_tree, "node_child")
        assert updated_child is not None
        assert updated_child.summary.startswith("Liabilities:")
    finally:
        store.close()
