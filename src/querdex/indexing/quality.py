from __future__ import annotations

from dataclasses import dataclass

from querdex.schemas import Section, TreeNode
from querdex.utils.tree_ops import walk_nodes


@dataclass(frozen=True)
class TreeQualityReport:
    coverage_ratio: float
    boundary_coherence: float
    max_depth: int
    avg_depth: float
    total_nodes: int


class TreeQualityEvaluator:
    """Simple tree-quality checks used for evaluation harness and CI."""

    def evaluate(self, root: TreeNode, sections: list[Section]) -> TreeQualityReport:
        pages = {section.page_number for section in sections}
        covered_pages: set[int] = set()
        depths: list[int] = []

        internal_nodes = 0
        coherent_internal = 0
        for node in walk_nodes(root):
            depths.append(node.depth)
            for page in range(node.start_page, node.end_page + 1):
                covered_pages.add(page)
            if node.children:
                internal_nodes += 1
                if self._is_boundary_coherent(node):
                    coherent_internal += 1

        coverage_ratio = len(covered_pages & pages) / max(1, len(pages))
        boundary_coherence = coherent_internal / max(1, internal_nodes)
        total_nodes = len(depths)
        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / max(1, total_nodes)

        return TreeQualityReport(
            coverage_ratio=coverage_ratio,
            boundary_coherence=boundary_coherence,
            max_depth=max_depth,
            avg_depth=avg_depth,
            total_nodes=total_nodes,
        )

    @staticmethod
    def _is_boundary_coherent(node: TreeNode) -> bool:
        if not node.children:
            return True
        ordered = sorted(node.children, key=lambda child: child.start_page)
        return all(left.end_page <= right.start_page for left, right in zip(ordered, ordered[1:], strict=False))
