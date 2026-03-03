from __future__ import annotations

import hashlib
from dataclasses import dataclass

from querdex.indexing.tree_builder import AdaptiveTreeBuilder
from querdex.schemas import Section, TreeNode
from querdex.utils.tree_ops import walk_nodes


@dataclass(frozen=True)
class DiffResult:
    changed_section_ids: set[str]
    changed_pages: set[int]
    added_section_ids: set[str]
    removed_section_ids: set[str]


def section_hash_map(sections: list[Section]) -> dict[str, str]:
    result: dict[str, str] = {}
    for section in sections:
        payload = f"{section.page_number}|{section.content}|{section.metadata}"
        digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
        result[section.section_id] = digest
    return result


def compute_section_diff(old_sections: list[Section], new_sections: list[Section]) -> DiffResult:
    old_hash = section_hash_map(old_sections)
    new_hash = section_hash_map(new_sections)

    old_ids = set(old_hash)
    new_ids = set(new_hash)
    added = new_ids - old_ids
    removed = old_ids - new_ids

    common = old_ids & new_ids
    changed = {sid for sid in common if old_hash[sid] != new_hash[sid]}

    new_by_id = {section.section_id: section for section in new_sections}
    old_by_id = {section.section_id: section for section in old_sections}

    changed_pages = {new_by_id[sid].page_number for sid in added | changed if sid in new_by_id}
    changed_pages.update({old_by_id[sid].page_number for sid in removed if sid in old_by_id})

    return DiffResult(
        changed_section_ids=changed,
        changed_pages=changed_pages,
        added_section_ids=added,
        removed_section_ids=removed,
    )


def map_changed_pages_to_nodes(root: TreeNode, changed_pages: set[int]) -> set[str]:
    affected: set[str] = set()
    for node in walk_nodes(root):
        page_range = set(range(node.start_page, node.end_page + 1))
        if page_range & changed_pages:
            affected.add(node.node_id)
    return affected


def partial_rebuild_tree(
    *,
    tree_builder: AdaptiveTreeBuilder,
    existing_root: TreeNode,
    all_sections: list[Section],
    changed_pages: set[int],
    doc_id: str,
    title: str,
) -> TreeNode:
    if not changed_pages:
        return existing_root

    all_sections_sorted = sorted(all_sections, key=lambda s: (s.page_number, s.section_id))
    if not existing_root.children:
        return tree_builder.build(doc_id, all_sections_sorted, title)

    rebuilt_children: list[TreeNode] = []
    for child in existing_root.children:
        child_pages = set(range(child.start_page, child.end_page + 1))
        intersects = bool(child_pages & changed_pages)
        if not intersects:
            rebuilt_children.append(child)
            continue

        subset = [
            section
            for section in all_sections_sorted
            if child.start_page <= section.page_number <= child.end_page
        ]
        if not subset:
            continue

        rebuilt = tree_builder.build(doc_id, subset, child.title)
        replacement_children = rebuilt.children if rebuilt.children else [rebuilt]
        # Keep one representative child node to replace this branch.
        replacement = replacement_children[0]
        replacement.depth = child.depth
        replacement.node_id = child.node_id
        rebuilt_children.append(replacement)

    new_root = tree_builder.build(doc_id, all_sections_sorted, title)
    if rebuilt_children:
        new_root.children = rebuilt_children
    new_root.node_id = existing_root.node_id
    new_root.depth = existing_root.depth
    return new_root
