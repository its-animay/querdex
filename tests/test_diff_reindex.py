from __future__ import annotations

from querdex.indexing import (
    AdaptiveTreeBuilder,
    compute_section_diff,
    map_changed_pages_to_nodes,
    partial_rebuild_tree,
    section_hash_map,
)
from querdex.schemas import Section, TreeNode


def _section(section_id: str, page: int, content: str) -> Section:
    return Section(
        section_id=section_id,
        doc_id="doc_diff",
        content=content,
        page_number=page,
        source_format="text",
        metadata={},
    )


def test_section_diff_detects_changed_added_removed() -> None:
    old_sections = [
        _section("sec_0001", 1, "Revenue 100"),
        _section("sec_0002", 2, "Liabilities 40"),
    ]
    new_sections = [
        _section("sec_0001", 1, "Revenue 100"),
        _section("sec_0002", 2, "Liabilities 50"),
        _section("sec_0003", 3, "Cash flow 20"),
    ]

    diff = compute_section_diff(old_sections, new_sections)

    assert diff.changed_section_ids == {"sec_0002"}
    assert diff.added_section_ids == {"sec_0003"}
    assert diff.removed_section_ids == set()
    assert diff.changed_pages == {2, 3}


def test_map_changed_pages_to_nodes_returns_overlapping_nodes() -> None:
    root = TreeNode(
        node_id="root",
        doc_id="doc_diff",
        title="Root",
        summary="all",
        start_page=1,
        end_page=4,
        depth=0,
        children=[
            TreeNode(
                node_id="n1",
                doc_id="doc_diff",
                title="One",
                summary="page one",
                start_page=1,
                end_page=1,
                depth=1,
            ),
            TreeNode(
                node_id="n2",
                doc_id="doc_diff",
                title="Two Three",
                summary="page two and three",
                start_page=2,
                end_page=3,
                depth=1,
            ),
        ],
    )

    affected = map_changed_pages_to_nodes(root, {3})
    assert "root" in affected
    assert "n2" in affected
    assert "n1" not in affected


def test_partial_rebuild_preserves_root_identity() -> None:
    sections = [
        _section("sec_0001", 1, "Revenue 100"),
        _section("sec_0002", 2, "Liabilities 40"),
        _section("sec_0003", 3, "Cash flow 20"),
    ]
    builder = AdaptiveTreeBuilder()
    root = builder.build("doc_diff", sections, "Finance")
    hashes_before = section_hash_map(sections)

    updated_sections = [
        _section("sec_0001", 1, "Revenue 100"),
        _section("sec_0002", 2, "Liabilities 55"),
        _section("sec_0003", 3, "Cash flow 20"),
    ]
    hashes_after = section_hash_map(updated_sections)
    changed_pages = {2} if hashes_before != hashes_after else set()

    rebuilt = partial_rebuild_tree(
        tree_builder=builder,
        existing_root=root,
        all_sections=updated_sections,
        changed_pages=changed_pages,
        doc_id="doc_diff",
        title="Finance",
    )

    assert rebuilt.node_id == root.node_id
    assert rebuilt.start_page == 1
    assert rebuilt.end_page == 3
