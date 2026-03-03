from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from querdex.indexing.entity_extractor import EntityExtractor
from querdex.indexing.graph_builder import KnowledgeGraphBuilder
from querdex.indexing.tree_builder import AdaptiveTreeBuilder
from querdex.schemas import IndexingResult, Section
from querdex.utils import validate_llm_payload

if TYPE_CHECKING:
    from querdex.llm import LLMClient


@dataclass(frozen=True)
class TokenBudget:
    tree_budget: int = 120_000
    entity_budget: int = 80_000
    graph_budget: int = 80_000


class IndexBuilderCoordinator:
    """Single entrypoint for indexing that runs tree/entity/graph builders in parallel."""

    def __init__(
        self,
        tree_builder: AdaptiveTreeBuilder | None = None,
        entity_extractor: EntityExtractor | None = None,
        graph_builder: KnowledgeGraphBuilder | None = None,
        budgets: TokenBudget | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._tree_builder = tree_builder or AdaptiveTreeBuilder(llm_client=llm_client)
        self._entity_extractor = entity_extractor or EntityExtractor()
        self._graph_builder = graph_builder or KnowledgeGraphBuilder()
        self._budgets = budgets or TokenBudget()

    async def build(self, doc_id: str, title: str, sections: list[Section]) -> IndexingResult:
        self._check_budget(sections)

        tree_task = asyncio.to_thread(self._tree_builder.build, doc_id, sections, title)
        entity_pre_task = asyncio.to_thread(self._entity_extractor.pre_extract, sections)
        tree_root, (section_entities, section_xrefs) = await asyncio.gather(tree_task, entity_pre_task)

        graph_pre_task = asyncio.to_thread(self._graph_builder.prebuild, tree_root)
        entity_task = asyncio.to_thread(
            self._entity_extractor.finalize,
            doc_id,
            sections,
            tree_root,
            section_entities,
            section_xrefs,
        )
        graph_draft, (entity_map, cross_refs) = await asyncio.gather(graph_pre_task, entity_task)
        graph_nodes, graph_edges = self._graph_builder.finalize(graph_draft, entity_map, cross_refs)

        result = IndexingResult(
            tree_root=tree_root,
            entity_map=entity_map,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )
        # HI-032b: all builder outputs must conform to strict schemas.
        validated = validate_llm_payload(IndexingResult, result.model_dump())
        return validated

    def _check_budget(self, sections: list[Section]) -> None:
        estimated_tokens = sum(max(1, len(s.content.split())) for s in sections)
        if estimated_tokens > self._budgets.tree_budget:
            msg = f"Input estimated at {estimated_tokens} tokens exceeds tree budget {self._budgets.tree_budget}"
            raise ValueError(msg)
        if estimated_tokens > self._budgets.entity_budget:
            msg = f"Input estimated at {estimated_tokens} tokens exceeds entity budget {self._budgets.entity_budget}"
            raise ValueError(msg)
        if estimated_tokens > self._budgets.graph_budget:
            msg = f"Input estimated at {estimated_tokens} tokens exceeds graph budget {self._budgets.graph_budget}"
            raise ValueError(msg)
