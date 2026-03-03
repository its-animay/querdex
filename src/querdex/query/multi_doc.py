from __future__ import annotations

from copy import deepcopy

from querdex.schemas import TreeNode


def build_virtual_super_tree(doc_roots: list[tuple[str, TreeNode]]) -> TreeNode:
    """Create synthetic root with each document root as a child (HI-077)."""
    if not doc_roots:
        msg = "No document roots provided for multi-document merge"
        raise ValueError(msg)

    min_page = min(root.start_page for _doc_id, root in doc_roots)
    max_page = max(root.end_page for _doc_id, root in doc_roots)

    children: list[TreeNode] = []
    for idx, (doc_id, root) in enumerate(doc_roots, start=1):
        cloned = deepcopy(root)
        cloned.title = f"{cloned.title} [{doc_id}]"
        cloned.depth = 1
        cloned.node_id = f"super_{idx:04d}_{doc_id}"
        children.append(cloned)

    return TreeNode(
        node_id="synthetic_root",
        doc_id="multi_doc",
        title="Synthetic Multi-Document Root",
        summary="Virtual super-tree root for comparative retrieval across documents.",
        start_page=min_page,
        end_page=max_page,
        depth=0,
        children=children,
    )
