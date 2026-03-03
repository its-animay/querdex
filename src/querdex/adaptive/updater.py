from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from querdex.schemas import QueryResult, TreeNode
from querdex.storage import SQLiteStore
from querdex.utils import compute_query_cluster_id, extract_json, find_node, walk_nodes

if TYPE_CHECKING:
    from querdex.llm import LLMClient


@dataclass(frozen=True)
class MisleadingNode:
    node_id: str
    visit_rate: float
    utility_rate: float


class AdaptiveUpdater:
    """Updates node visit/utility/affinity signals and performs summary regeneration."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client

    def update_tree(
        self,
        *,
        doc_id: str,
        tree_root: TreeNode,
        query_results: list[QueryResult],
        feedback_events: list[dict[str, object]],
        store: SQLiteStore,
    ) -> tuple[TreeNode, list[MisleadingNode]]:
        doc_results = [result for result in query_results if any(src.doc_id == doc_id for src in result.source_nodes)]
        doc_feedback = [event for event in feedback_events if event.get("doc_id") == doc_id]
        total_queries = max(1, len(doc_feedback))

        visit_counter: Counter[str] = Counter()
        use_counter: Counter[str] = Counter()

        for event in doc_feedback:
            for node_id in self._as_str_list(event.get("visited_nodes")):
                visit_counter[str(node_id)] += 1
            for node_id in self._as_str_list(event.get("used_nodes")):
                use_counter[str(node_id)] += 1

        cluster_hits: dict[str, Counter[str]] = defaultdict(Counter)
        cluster_totals: Counter[str] = Counter()
        for result in doc_results:
            cluster = compute_query_cluster_id(result.rewritten_query)
            cluster_totals[cluster] += 1
            for src in result.source_nodes:
                if src.doc_id == doc_id:
                    cluster_hits[src.node_id][cluster] += 1

        misleading: list[MisleadingNode] = []
        for node in walk_nodes(tree_root):
            visits = visit_counter.get(node.node_id, 0)
            uses = use_counter.get(node.node_id, 0)
            node.visit_rate = visits / total_queries
            node.utility_rate = uses / max(1, visits)
            store.upsert_node_metric(doc_id=doc_id, node_id=node.node_id, visited=visits, used=uses)

            for cluster_id, hit_count in cluster_hits.get(node.node_id, {}).items():
                score = hit_count / max(1, cluster_totals[cluster_id])
                node.affinity_scores[cluster_id] = score
                store.upsert_affinity_score(doc_id=doc_id, node_id=node.node_id, cluster_id=cluster_id, score=score)

            if node.visit_rate >= 0.6 and node.utility_rate <= 0.2:
                misleading.append(
                    MisleadingNode(
                        node_id=node.node_id,
                        visit_rate=node.visit_rate,
                        utility_rate=node.utility_rate,
                    )
                )
                store.enqueue_summary_regen(
                    doc_id=doc_id,
                    node_id=node.node_id,
                    prompt=(
                        f"Rewrite summary for node {node.node_id} to strictly describe pages "
                        f"{node.start_page}-{node.end_page}."
                    ),
                )

        updated = self._process_summary_regeneration(doc_id=doc_id, tree_root=tree_root, store=store)
        return updated, misleading

    def _process_summary_regeneration(self, *, doc_id: str, tree_root: TreeNode, store: SQLiteStore) -> TreeNode:
        queue = store.pending_summary_regen(limit=20)
        for item in queue:
            if item.get("doc_id") != doc_id:
                continue
            node_id = str(item["node_id"])
            node = find_node(tree_root, node_id)
            if node is None:
                continue
            sections = store.fetch_sections_by_page_range(doc_id, node.start_page, node.end_page)
            combined = " ".join(section.content for section in sections)
            rewritten = self._rewrite_summary(node.title, combined, node.start_page, node.end_page)
            node.summary = rewritten
            store.mark_summary_regen_done(queue_id=int(item["queue_id"]), new_summary=rewritten)
        return tree_root

    def _rewrite_summary(self, title: str, content: str, start_page: int = 0, end_page: int = 0) -> str:
        if self._llm is not None:
            content_preview = " ".join(content.split())[:800]
            user_prompt = (
                f"Title: {title}\n"
                f"Pages: {start_page}-{end_page}\n"
                f"Content:\n{content_preview}\n\n"
                "Rewrite the summary to describe ONLY what is literally in these pages. "
                "Do not mention topics not present. "
                'Output JSON only: {"summary": "..."}'
            )
            try:
                raw = self._llm.complete(
                    system="You rewrite node summaries to accurately reflect only what is in the content.",
                    user=user_prompt,
                    model=self._llm.tier1_model,
                    max_tokens=256,
                )
                data = json.loads(extract_json(raw))
                summary = str(data.get("summary", "")).strip()
                if summary:
                    return summary
            except Exception:  # noqa: BLE001
                pass  # fall through to heuristic

        # Heuristic fallback
        normalized = " ".join(content.split())
        snippet = normalized[:240] if normalized else "No textual content available"
        return f"{title}: {snippet}"

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]
