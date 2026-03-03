from __future__ import annotations

from querdex.indexing import TreeQualityEvaluator
from querdex.schemas import Section, TreeNode


def test_tree_quality_evaluator_reports_coverage_and_depth() -> None:
    sections = [
        Section(
            section_id="sec_0001",
            doc_id="doc_quality",
            content="Page one content",
            page_number=1,
            source_format="text",
            metadata={},
        ),
        Section(
            section_id="sec_0002",
            doc_id="doc_quality",
            content="Page two content",
            page_number=2,
            source_format="text",
            metadata={},
        ),
    ]
    tree = TreeNode(
        node_id="root",
        doc_id="doc_quality",
        title="Root",
        summary="summary",
        start_page=1,
        end_page=2,
        depth=0,
        children=[
            TreeNode(
                node_id="leaf1",
                doc_id="doc_quality",
                title="Leaf 1",
                summary="s1",
                start_page=1,
                end_page=1,
                depth=1,
            ),
            TreeNode(
                node_id="leaf2",
                doc_id="doc_quality",
                title="Leaf 2",
                summary="s2",
                start_page=2,
                end_page=2,
                depth=1,
            ),
        ],
    )

    report = TreeQualityEvaluator().evaluate(tree, sections)
    assert report.coverage_ratio == 1.0
    assert report.boundary_coherence == 1.0
    assert report.max_depth == 1
    assert report.total_nodes == 3
