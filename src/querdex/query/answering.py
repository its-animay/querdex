from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from querdex.schemas import SearchCandidate, SourceNode, TreeNode
from querdex.storage import SQLiteStore
from querdex.utils.llm_validation import extract_json
from querdex.utils.tree_ops import find_node

if TYPE_CHECKING:
    from querdex.llm import LLMClient


@dataclass
class RetrievedChunk:
    source: SourceNode
    text: str


class ContentRetriever:
    def retrieve(
        self,
        *,
        doc_id: str,
        tree_root: TreeNode,
        candidates: list[SearchCandidate],
        store: SQLiteStore,
    ) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for candidate in candidates:
            node = find_node(tree_root, candidate.node_id)
            if node is None:
                continue
            sections = store.fetch_sections_by_page_range(doc_id, node.start_page, node.end_page)
            text = "\n".join(section.content for section in sections).strip()
            if not text:
                continue
            source = SourceNode(
                node_id=node.node_id,
                doc_id=doc_id,
                title=node.title,
                pages=f"{node.start_page}-{node.end_page}",
                confidence=candidate.confidence,
            )
            chunks.append(RetrievedChunk(source=source, text=text))
        return chunks


class AnswerGenerator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, float, list[SourceNode]]:
        if not chunks:
            return "No relevant content found in the indexed document.", 0.0, []

        top_chunks = chunks[:5]
        sources = [c.source for c in top_chunks]

        if self._llm is not None:
            section_texts = "\n\n".join(
                f"[Source: {c.source.title}, pages {c.source.pages}]\n{c.text}"
                for c in top_chunks
            )
            user_prompt = (
                f"Query: {query}\n\n"
                f"Relevant sections (in retrieval order):\n{section_texts}\n\n"
                "Write a direct, concise answer with inline source citations (title + pages). "
                'Output JSON only: {"answer": "...", "confidence": 0.0}'
            )
            try:
                raw = self._llm.complete(
                    system="You are a precise document Q&A assistant. Always cite sources.",
                    user=user_prompt,
                    model=self._llm.tier2_model,
                    max_tokens=512,
                )
                data = json.loads(extract_json(raw))
                answer = str(data.get("answer", "")).strip()
                confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
                if answer:
                    return answer, confidence, sources
            except Exception:  # noqa: BLE001
                pass  # fall through to heuristic

        # Heuristic fallback
        answer_lines = [f"Query: {query}", "", "Answer summary:"]
        for chunk in top_chunks:
            snippet = " ".join(chunk.text.split())[:260]
            answer_lines.append(f"- {snippet} (Source: {chunk.source.title}, pages {chunk.source.pages})")

        confidence = sum(c.source.confidence for c in top_chunks) / len(top_chunks)
        return "\n".join(answer_lines), confidence, sources
