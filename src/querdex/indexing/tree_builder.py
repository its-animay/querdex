from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

from querdex.schemas import Section, TreeNode
from querdex.utils.llm_validation import extract_json

if TYPE_CHECKING:
    from querdex.llm import LLMClient


@dataclass(frozen=True)
class TreeBuilderConfig:
    max_tokens_per_node: int = 280
    min_sections_per_group: int = 2
    boundary_similarity_threshold: float = 0.18
    max_sections_per_leaf_group: int = 4


class AdaptiveTreeBuilder:
    """Adaptive semantic tree builder with recursive token-constrained splitting."""

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(
        self,
        config: TreeBuilderConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config or TreeBuilderConfig()
        self._llm = llm_client
        self._next_node_index = 0

    def build(self, doc_id: str, sections: list[Section], title: str) -> TreeNode:
        ordered_sections = sorted(sections, key=lambda s: (s.page_number, s.section_id))
        if not ordered_sections:
            msg = "Cannot build tree from empty section list"
            raise ValueError(msg)

        self._next_node_index = 0

        root = TreeNode(
            node_id=self._alloc_node_id(),
            doc_id=doc_id,
            title=title,
            summary=self._build_summary(ordered_sections, section_label=title),
            start_page=min(s.page_number for s in ordered_sections),
            end_page=max(s.page_number for s in ordered_sections),
            depth=0,
            children=[],
        )

        initial_groups = self._detect_semantic_boundaries(ordered_sections)
        for idx, group in enumerate(initial_groups, start=1):
            group_title = self._infer_group_title(group, fallback=f"Topic Group {idx}")
            child = self._build_subtree(
                doc_id=doc_id,
                sections=group,
                depth=1,
                group_title=group_title,
            )
            root.children.append(child)

        return root

    def _build_subtree(
        self,
        *,
        doc_id: str,
        sections: list[Section],
        depth: int,
        group_title: str,
    ) -> TreeNode:
        section_tokens = self._count_tokens(sections)
        node = TreeNode(
            node_id=self._alloc_node_id(),
            doc_id=doc_id,
            title=group_title,
            summary=self._build_summary(sections, section_label=group_title),
            start_page=min(s.page_number for s in sections),
            end_page=max(s.page_number for s in sections),
            depth=depth,
            children=[],
        )

        if len(sections) == 1:
            return node

        should_split = (
            section_tokens > self.config.max_tokens_per_node or len(sections) > self.config.max_sections_per_leaf_group
        )

        if not should_split:
            node.children = [
                self._leaf_from_section(doc_id=doc_id, section=section, depth=depth + 1) for section in sections
            ]
            return node

        child_groups = self._detect_semantic_boundaries(sections)
        if len(child_groups) <= 1:
            child_groups = self._hard_split(sections)

        for idx, group in enumerate(child_groups, start=1):
            title = self._infer_group_title(group, fallback=f"{group_title} - Part {idx}")
            child = self._build_subtree(
                doc_id=doc_id,
                sections=group,
                depth=depth + 1,
                group_title=title,
            )
            node.children.append(child)

        return node

    def _leaf_from_section(self, *, doc_id: str, section: Section, depth: int) -> TreeNode:
        heading = str(section.metadata.get("heading") or "")
        leaf_title = heading or f"Section {section.section_id}"
        return TreeNode(
            node_id=self._alloc_node_id(),
            doc_id=doc_id,
            title=leaf_title,
            summary=self._build_summary([section], section_label=leaf_title),
            start_page=section.page_number,
            end_page=section.page_number,
            depth=depth,
            children=[],
        )

    def _detect_semantic_boundaries(self, sections: list[Section]) -> list[list[Section]]:
        if len(sections) <= 1:
            return [sections]

        groups: list[list[Section]] = []
        active_group: list[Section] = [sections[0]]

        for prev, cur in zip(sections, sections[1:], strict=False):
            similarity = self._similarity(prev.content, cur.content)
            same_heading = self._heading(prev) == self._heading(cur)
            active_group_tokens = self._count_tokens(active_group)
            exceeds_budget = active_group_tokens >= self.config.max_tokens_per_node

            boundary = (
                similarity < self.config.boundary_similarity_threshold
                and len(active_group) >= self.config.min_sections_per_group
                and not same_heading
            ) or exceeds_budget

            if boundary:
                groups.append(active_group)
                active_group = [cur]
            else:
                active_group.append(cur)

        if active_group:
            groups.append(active_group)
        return groups

    def _hard_split(self, sections: list[Section]) -> list[list[Section]]:
        midpoint = max(1, len(sections) // 2)
        left = sections[:midpoint]
        right = sections[midpoint:]
        if not left or not right:
            return [sections]
        return [left, right]

    def _build_summary(self, sections: list[Section], section_label: str) -> str:
        page_start = min(section.page_number for section in sections)
        page_end = max(section.page_number for section in sections)

        if self._llm is not None:
            text = " ".join(section.content for section in sections)
            content_preview = " ".join(text.split())[:1200]
            user_prompt = (
                f"Title: {section_label}\n"
                f"Content (pages {page_start}-{page_end}):\n{content_preview}\n\n"
                "Write 1-2 sentences that would help an LLM decide if this section answers a query. "
                'Output JSON only: {"summary": "..."}'
            )
            try:
                raw = self._llm.complete(
                    system="You generate concise retrieval-optimized document summaries.",
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
        text = " ".join(section.content for section in sections)
        normalized = " ".join(text.split())
        preview = normalized[:220] if normalized else "No content"
        keywords = ", ".join(islice(self._keyword_candidates(normalized), 4))
        if keywords:
            return f"{section_label}: covers pages {page_start}-{page_end}; keywords: {keywords}. Summary: {preview}"
        return f"{section_label}: covers pages {page_start}-{page_end}. Summary: {preview}"

    def _keyword_candidates(self, text: str) -> list[str]:
        counts: dict[str, int] = {}
        for token in self._TOKEN_RE.findall(text.lower()):
            if len(token) < 4:
                continue
            counts[token] = counts.get(token, 0) + 1
        return [word for word, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)]

    def _similarity(self, left: str, right: str) -> float:
        a = set(self._TOKEN_RE.findall(left.lower()))
        b = set(self._TOKEN_RE.findall(right.lower()))
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _heading(section: Section) -> str:
        return str(section.metadata.get("heading") or "").strip().lower()

    @staticmethod
    def _count_tokens(sections: list[Section]) -> int:
        return sum(max(1, len(section.content.split())) for section in sections)

    def _alloc_node_id(self) -> str:
        node_id = f"node_{self._next_node_index:04d}"
        self._next_node_index += 1
        return node_id

    @staticmethod
    def _infer_group_title(sections: list[Section], fallback: str) -> str:
        headings = [str(section.metadata.get("heading") or "").strip() for section in sections]
        for heading in headings:
            if heading:
                return heading
        return fallback
