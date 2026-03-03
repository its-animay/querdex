from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from querdex.schemas import QueryAnalysis, SearchCandidate, TreeNode
from querdex.utils.llm_validation import extract_json
from querdex.utils.tree_ops import find_node

if TYPE_CHECKING:
    from querdex.llm import LLMClient

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class SearchRun:
    candidates: list[SearchCandidate]
    traversal_path: list[str]
    tier1_calls: int
    tier2_calls: int
    cache_hit: bool


class TieredSearchEngine:
    def __init__(
        self,
        affinity_threshold: float = 0.8,
        tier2_threshold: float = 0.25,
        max_tier2_nodes: int = 80,
        tier1_client: LLMClient | None = None,
        tier2_client: LLMClient | None = None,
    ) -> None:
        self.affinity_threshold = affinity_threshold
        self.tier2_threshold = tier2_threshold
        self.max_tier2_nodes = max_tier2_nodes
        self._tier1_llm = tier1_client
        self._tier2_llm = tier2_client

    def run(
        self,
        *,
        tree_root: TreeNode,
        analysis: QueryAnalysis,
        cached: list[dict[str, Any]] | None,
        entity_seed_nodes: set[str],
    ) -> SearchRun:
        if cached:
            candidates = [
                SearchCandidate(
                    node_id=str(item["node_id"]),
                    confidence=float(item["confidence"]),
                )
                for item in cached
            ]
            traversal = [item.node_id for item in candidates]
            return SearchRun(candidates, traversal, tier1_calls=0, tier2_calls=0, cache_hit=True)

        root_candidates = tree_root.children if tree_root.children else [tree_root]

        # HI-072b: affinity fast-path nodes skip Tier 1.
        fast_path: list[TreeNode] = []
        regular_path: list[TreeNode] = []
        cluster = analysis.query_cluster_id
        for node in root_candidates:
            affinity = node.affinity_scores.get(cluster, 0.0)
            if affinity >= self.affinity_threshold:
                fast_path.append(node)
            else:
                regular_path.append(node)

        if entity_seed_nodes:
            regular_path = [n for n in regular_path if self._matches_entity_seed(n, tree_root, entity_seed_nodes)]
            fast_path = [n for n in fast_path if self._matches_entity_seed(n, tree_root, entity_seed_nodes)]

        # HI-072: single batched Tier 1 evaluation over root candidates.
        tier1_scores = self._tier1_batch_score(analysis.rewritten_query, regular_path)
        tier1_selected = [node for node, score in tier1_scores if score >= 0.1]
        if not tier1_selected and tier1_scores:
            tier1_selected = [node for node, _score in sorted(tier1_scores, key=lambda item: item[1], reverse=True)[:3]]

        selected = fast_path + tier1_selected

        traversal_path: list[str] = [n.node_id for n in selected]
        tier2_candidates = self._tier2_rank(analysis.rewritten_query, selected)
        traversal_path.extend([c.node_id for c in tier2_candidates])

        return SearchRun(
            candidates=tier2_candidates,
            traversal_path=traversal_path,
            tier1_calls=1 if regular_path else 0,
            tier2_calls=len(selected),
            cache_hit=False,
        )

    def _tier1_batch_score(self, query: str, nodes: list[TreeNode]) -> list[tuple[TreeNode, float]]:
        if self._tier1_llm is not None and nodes:
            section_lines = "\n".join(
                f"[{i}] {node.title}: {node.summary}" for i, node in enumerate(nodes)
            )
            user_prompt = (
                f"Query: {query}\n\n"
                "For each section below, answer YES if it likely contains the answer, else NO.\n"
                f"{section_lines}\n\n"
                'Output JSON array only: [{"index": 0, "relevant": true}, ...]'
            )
            try:
                raw = self._tier1_llm.complete(
                    system="You are a document retrieval filter. Answer YES or NO per section.",
                    user=user_prompt,
                    model=self._tier1_llm.tier1_model,
                    max_tokens=512,
                )
                items = json.loads(extract_json(raw))
                index_to_relevant: dict[int, bool] = {
                    int(item["index"]): bool(item.get("relevant", False))
                    for item in items
                    if isinstance(item, dict) and "index" in item
                }
                return [(node, 1.0 if index_to_relevant.get(i, False) else 0.0) for i, node in enumerate(nodes)]
            except Exception:  # noqa: BLE001
                pass  # fall through to heuristic

        # Heuristic fallback
        query_tokens = set(_WORD_RE.findall(query.lower()))
        scored: list[tuple[TreeNode, float]] = []
        for node in nodes:
            node_text = f"{node.title} {node.summary}".lower()
            node_tokens = set(_WORD_RE.findall(node_text))
            overlap = len(query_tokens & node_tokens)
            score = overlap / max(len(query_tokens), 1)
            scored.append((node, score))
        return scored

    def _tier2_rank(self, query: str, nodes: list[TreeNode]) -> list[SearchCandidate]:
        if self._tier2_llm is not None and nodes:
            results: list[SearchCandidate] = []
            queue = list(nodes)
            visited: set[str] = set()
            inspected = 0

            while queue and inspected < self.max_tier2_nodes:
                node = queue.pop(0)
                if node.node_id in visited:
                    continue
                visited.add(node.node_id)
                inspected += 1

                user_prompt = (
                    f"Query: {query}\n\n"
                    f"Section Title: {node.title}\n"
                    f"Section Summary: {node.summary}\n"
                    f"Pages: {node.start_page}-{node.end_page}\n\n"
                    "Is this section relevant to the query? "
                    'Output JSON only: {"relevant": true, "confidence": 0.0, "explore_children": []}'
                )
                try:
                    raw = self._tier2_llm.complete(
                        system="You are a document retrieval expert. Reason carefully about relevance.",
                        user=user_prompt,
                        model=self._tier2_llm.tier2_model,
                        max_tokens=256,
                    )
                    data = json.loads(extract_json(raw))
                    relevant = bool(data.get("relevant", False))
                    confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
                    explore_children: list[str] = [str(c) for c in data.get("explore_children", [])]
                except Exception:  # noqa: BLE001
                    relevant = False
                    confidence = 0.0
                    explore_children = []

                if relevant and confidence >= self.tier2_threshold:
                    results.append(SearchCandidate(node_id=node.node_id, confidence=confidence))
                    # Enqueue children explicitly requested by LLM
                    requested = {c.node_id: c for c in node.children if c.node_id in explore_children}
                    queue[:0] = list(requested.values())

            deduped: dict[str, SearchCandidate] = {}
            for candidate in results:
                current = deduped.get(candidate.node_id)
                if current is None or candidate.confidence > current.confidence:
                    deduped[candidate.node_id] = candidate
            return sorted(deduped.values(), key=lambda c: c.confidence, reverse=True)[:8]

        # Heuristic fallback
        query_tokens = set(_WORD_RE.findall(query.lower()))
        heuristic_results: list[SearchCandidate] = []
        scored_nodes: list[SearchCandidate] = []
        h_queue = list(nodes)
        h_visited: set[str] = set()
        h_inspected = 0

        while h_queue and h_inspected < self.max_tier2_nodes:
            node = h_queue.pop(0)
            if node.node_id in h_visited:
                continue
            h_visited.add(node.node_id)
            h_inspected += 1

            best_score = self._score_node(node, query_tokens)
            scored_nodes.append(SearchCandidate(node_id=node.node_id, confidence=min(1.0, best_score)))
            if best_score >= self.tier2_threshold:
                heuristic_results.append(SearchCandidate(node_id=node.node_id, confidence=min(1.0, best_score)))
                h_queue.extend(node.children)
                continue

            title_overlap = len(query_tokens & set(_WORD_RE.findall(node.title.lower())))
            if title_overlap > 0:
                h_queue.extend(node.children)

        h_deduped: dict[str, SearchCandidate] = {}
        source = (
            heuristic_results
            if heuristic_results
            else sorted(scored_nodes, key=lambda c: c.confidence, reverse=True)[:3]
        )
        for candidate in source:
            current = h_deduped.get(candidate.node_id)
            if current is None or candidate.confidence > current.confidence:
                h_deduped[candidate.node_id] = candidate

        return sorted(h_deduped.values(), key=lambda c: c.confidence, reverse=True)[:8]

    @staticmethod
    def _score_node(node: TreeNode, query_tokens: set[str]) -> float:
        node_tokens = set(_WORD_RE.findall(f"{node.title} {node.summary}".lower()))
        overlap = len(query_tokens & node_tokens)
        if not node_tokens:
            return 0.0
        return overlap / max(len(query_tokens), 1)

    @staticmethod
    def _matches_entity_seed(node: TreeNode, tree_root: TreeNode, seed_nodes: set[str]) -> bool:
        if node.node_id in seed_nodes:
            return True
        for child in node.children:
            if child.node_id in seed_nodes:
                return True
            leaf = find_node(tree_root, child.node_id)
            if leaf is not None and leaf.node_id in seed_nodes:
                return True
        return False
