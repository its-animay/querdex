from __future__ import annotations

from collections.abc import Iterable

from querdex.schemas import TreeNode


def walk_nodes(root: TreeNode) -> Iterable[TreeNode]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def find_node(root: TreeNode, node_id: str) -> TreeNode | None:
    for node in walk_nodes(root):
        if node.node_id == node_id:
            return node
    return None


def compute_tree_stats(root: TreeNode) -> tuple[int, int, float]:
    depths = [node.depth for node in walk_nodes(root)]
    total_nodes = len(depths)
    max_depth = max(depths)
    avg_depth = sum(depths) / total_nodes
    return total_nodes, max_depth, avg_depth
