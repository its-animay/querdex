from __future__ import annotations

import re
from dataclasses import dataclass

from querdex.schemas import EntityRef, TreeNode
from querdex.utils import compute_query_embedding
from querdex.utils.tree_ops import walk_nodes


@dataclass(frozen=True)
class GraphDraft:
    nodes: list[tuple[str, str]]
    edges: list[tuple[str, str, str, float]]
    node_text: dict[str, str]


class KnowledgeGraphBuilder:
    """Builds graph nodes and edges from tree structure and extracted cross-references."""

    _TOKEN_RE = re.compile(r"[a-z0-9]+")
    _YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
    _CONTRADICT_TOKENS = {"not", "never", "no", "without", "decline", "decrease", "drop", "loss"}
    _POSITIVE_TOKENS = {"increase", "growth", "up", "gain", "profit", "improve"}

    def prebuild(self, tree_root: TreeNode) -> GraphDraft:
        nodes: list[tuple[str, str]] = []
        edges: list[tuple[str, str, str, float]] = []
        node_text: dict[str, str] = {}

        for node in walk_nodes(tree_root):
            nodes.append((node.doc_id, node.node_id))
            node_text[node.node_id] = f"{node.title} {node.summary}"
            for child in node.children:
                edges.append((node.node_id, child.node_id, "ELABORATES", 0.95))

        return GraphDraft(nodes=nodes, edges=edges, node_text=node_text)

    def finalize(
        self,
        draft: GraphDraft,
        entity_map: dict[str, list[EntityRef]],
        cross_refs: dict[str, list[str]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str, float]]]:
        edges = list(draft.edges)

        # Build in-document REFERENCES edges from normalized xrefs.
        for source_node_id, references in cross_refs.items():
            for reference in references:
                targets = entity_map.get(reference, [])
                for target in targets:
                    if target.node_id == source_node_id:
                        continue
                    edges.append((source_node_id, target.node_id, "REFERENCES", 0.8))

        # Additional relation passes on in-document nodes.
        edges.extend(self._temporal_edges(draft))
        edges.extend(self._contradiction_edges(draft))

        return draft.nodes, self._dedupe_edges(edges)

    def build(
        self,
        tree_root: TreeNode,
        entity_map: dict[str, list[EntityRef]],
        cross_refs: dict[str, list[str]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str, float]]]:
        draft = self.prebuild(tree_root)
        return self.finalize(draft, entity_map, cross_refs)

    def cross_doc_similarity_edges(
        self,
        *,
        source_doc_id: str,
        source_nodes: list[tuple[str, str, str]],
        other_docs_nodes: list[tuple[str, str, str]],
        threshold: float = 0.45,
    ) -> list[tuple[str, str, str, float]]:
        """
        Build CROSS_DOC_SIMILAR edges.

        Input tuple format: (doc_id, node_id, text_summary).
        """
        edges: list[tuple[str, str, str, float]] = []
        source_vectors = {
            (doc_id, node_id): compute_query_embedding(text)
            for doc_id, node_id, text in source_nodes
            if doc_id == source_doc_id
        }
        other_vectors = {
            (doc_id, node_id): compute_query_embedding(text)
            for doc_id, node_id, text in other_docs_nodes
            if doc_id != source_doc_id
        }

        for s_doc, s_node, _s_text in source_nodes:
            if s_doc != source_doc_id:
                continue
            s_vector = source_vectors.get((s_doc, s_node))
            if s_vector is None:
                continue
            for o_doc, o_node, _o_text in other_docs_nodes:
                if o_doc == source_doc_id:
                    continue
                o_vector = other_vectors.get((o_doc, o_node))
                if o_vector is None:
                    continue
                score = self._cosine_similarity(s_vector, o_vector)
                if score >= threshold:
                    edges.append((f"{s_doc}:{s_node}", f"{o_doc}:{o_node}", "CROSS_DOC_SIMILAR", score))
        return self._dedupe_edges(edges)

    @staticmethod
    def _dedupe_edges(
        edges: list[tuple[str, str, str, float]],
    ) -> list[tuple[str, str, str, float]]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[tuple[str, str, str, float]] = []
        for source, target, edge_type, weight in edges:
            key = (source, target, edge_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((source, target, edge_type, weight))
        return deduped

    def _temporal_edges(self, draft: GraphDraft) -> list[tuple[str, str, str, float]]:
        ordered = [node_id for _doc_id, node_id in draft.nodes]
        with_year: list[tuple[int, str]] = []
        for node_id in ordered:
            years = [int(value) for value in self._YEAR_RE.findall(draft.node_text.get(node_id, ""))]
            if years:
                with_year.append((max(years), node_id))

        edges: list[tuple[str, str, str, float]] = []
        for left, right in zip(ordered, ordered[1:], strict=False):
            edges.append((left, right, "TEMPORALLY_FOLLOWS", 0.55))
        for (year_a, node_a), (year_b, node_b) in zip(with_year, with_year[1:], strict=False):
            if year_b >= year_a:
                edges.append((node_a, node_b, "TEMPORALLY_FOLLOWS", 0.8))
        return edges

    def _contradiction_edges(self, draft: GraphDraft) -> list[tuple[str, str, str, float]]:
        edges: list[tuple[str, str, str, float]] = []
        ordered_ids = [node_id for _doc_id, node_id in draft.nodes]
        for left, right in zip(ordered_ids, ordered_ids[1:], strict=False):
            left_tokens = self._tokenize(draft.node_text.get(left, ""))
            right_tokens = self._tokenize(draft.node_text.get(right, ""))
            shared = left_tokens & right_tokens
            if not shared:
                continue
            left_neg = bool(left_tokens & self._CONTRADICT_TOKENS)
            right_neg = bool(right_tokens & self._CONTRADICT_TOKENS)
            left_pos = bool(left_tokens & self._POSITIVE_TOKENS)
            right_pos = bool(right_tokens & self._POSITIVE_TOKENS)
            if (left_neg and right_pos) or (right_neg and left_pos):
                edges.append((left, right, "CONTRADICTS", 0.6))
        return edges

    def _tokenize(self, text: str) -> set[str]:
        tokens = {token for token in self._TOKEN_RE.findall(text.lower()) if len(token) >= 3}
        return tokens

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(lv * rv for lv, rv in zip(left, right, strict=True))
