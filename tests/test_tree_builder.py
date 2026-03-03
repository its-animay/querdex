from __future__ import annotations

from querdex.indexing.tree_builder import AdaptiveTreeBuilder, TreeBuilderConfig
from querdex.schemas import Section
from querdex.utils.tree_ops import walk_nodes


def _section(idx: int, heading: str, content: str) -> Section:
    return Section(
        section_id=f"sec_{idx:04d}",
        doc_id="doc_tree",
        content=content,
        page_number=idx,
        source_format="markdown",
        metadata={"heading": heading},
    )


def test_tree_builder_recursively_splits_large_groups() -> None:
    sections = [
        _section(1, "Finance", "Revenue growth margin operating cash flow debt leverage" * 4),
        _section(2, "Finance", "Revenue forecast margins and quarterly outlook" * 4),
        _section(3, "Operations", "Supply chain lead time logistics inventory" * 4),
        _section(4, "Operations", "Procurement bottlenecks and vendor risk" * 4),
        _section(5, "Risk", "Regulatory compliance risk legal exposure" * 4),
        _section(6, "Risk", "Cybersecurity controls breach response controls" * 4),
    ]

    builder = AdaptiveTreeBuilder(
        TreeBuilderConfig(
            max_tokens_per_node=25,
            boundary_similarity_threshold=0.10,
            max_sections_per_leaf_group=2,
        )
    )
    root = builder.build("doc_tree", sections, "Tree Test")

    assert root.children
    assert any(node.depth >= 2 for node in walk_nodes(root))
    assert root.start_page == 1
    assert root.end_page == 6
    assert "covers pages" in root.summary


def test_tree_builder_handles_single_section() -> None:
    sections = [_section(1, "Intro", "Single section content")]
    builder = AdaptiveTreeBuilder()
    root = builder.build("doc_tree", sections, "Single")

    assert len(root.children) == 1
    child = root.children[0]
    assert child.start_page == 1
    assert child.end_page == 1
