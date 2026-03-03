from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from querdex.adaptive import AdaptiveUpdater

if TYPE_CHECKING:
    from querdex.llm import LLMClient
from querdex.indexing import (
    EntityExtractor,
    IndexBuilderCoordinator,
    KnowledgeGraphBuilder,
    TreeQualityEvaluator,
    compute_section_diff,
    map_changed_pages_to_nodes,
    partial_rebuild_tree,
)
from querdex.ingestion import IngestionOrchestrator
from querdex.ops import HealthChecker, StructuredLogger, with_retry
from querdex.query import (
    AnswerGenerator,
    ContentRetriever,
    GraphWalker,
    QueryAnalyzer,
    QueryRouter,
    TieredSearchEngine,
    build_virtual_super_tree,
)
from querdex.schemas import (
    DocumentIndex,
    DocumentStats,
    EntityRef,
    IndexingResult,
    QueryAnalysis,
    QueryResult,
    SearchCandidate,
    Section,
    TreeNode,
)
from querdex.storage import SQLiteStore
from querdex.utils.tree_ops import compute_tree_stats, find_node, walk_nodes


class QuerdexEngine:
    _idempotency_seen: set[str] = set()
    _query_idempotency_cache: dict[str, QueryResult] = {}

    def __init__(self, store: SQLiteStore, llm_client: LLMClient | None = None) -> None:
        self.store = store
        self.ingestion = IngestionOrchestrator()
        self.coordinator = IndexBuilderCoordinator(llm_client=llm_client)
        self.entity_extractor = EntityExtractor()
        self.graph_builder = KnowledgeGraphBuilder()
        self.analyzer = QueryAnalyzer()
        self.router = QueryRouter()
        self.search = TieredSearchEngine(tier1_client=llm_client, tier2_client=llm_client)
        self.retriever = ContentRetriever()
        self.answer_generator = AnswerGenerator(llm_client=llm_client)
        self.graph_walker = GraphWalker()
        self.adaptive_updater = AdaptiveUpdater(llm_client=llm_client)
        self.tree_quality = TreeQualityEvaluator()
        self.health_checker = HealthChecker()
        self.logger = StructuredLogger("querdex.engine")

    async def index_document(self, file_path: str | Path, doc_id: str | None = None) -> DocumentIndex:
        path = Path(file_path)
        resolved_doc_id = doc_id or self._doc_id_from_path(path)
        title = path.stem.replace("_", " ").strip() or resolved_doc_id

        with self.logger.span("index_document", doc_id=resolved_doc_id, path=str(path)):
            idempotency_key = self._idempotency_key("index", resolved_doc_id, path)
            if idempotency_key in self._idempotency_seen:
                self.logger.info("index_idempotent_skip", doc_id=resolved_doc_id, path=str(path))
                version, tree = self.store.latest_tree(resolved_doc_id)
                return self._build_document_index(
                    doc_id=resolved_doc_id,
                    title=title,
                    source_format=path.suffix.lower().lstrip("."),
                    version=version,
                    tree=tree,
                    sections=self.store.sections_for_doc(resolved_doc_id),
                    entity_map={},
                )

            sections = self.ingestion.parse(path, resolved_doc_id)
            indexing_result = await self.coordinator.build(resolved_doc_id, title, sections)
            indexing_result = self._augment_cross_doc_edges(source_doc_id=resolved_doc_id, result=indexing_result)

            version = self._save_index_with_retry(
                doc_id=resolved_doc_id,
                title=title,
                source_format=path.suffix.lower().lstrip("."),
                sections=sections,
                indexing_result=indexing_result,
                event_type="index_build",
                event_details={"section_count": len(sections), "mode": "full"},
            )

            self._idempotency_seen.add(idempotency_key)
            report = self.tree_quality.evaluate(indexing_result.tree_root, sections)
            self.logger.info(
                "index_completed",
                doc_id=resolved_doc_id,
                version=version,
                nodes=report.total_nodes,
                coverage_ratio=report.coverage_ratio,
                boundary_coherence=report.boundary_coherence,
            )

            return self._build_document_index(
                doc_id=resolved_doc_id,
                title=title,
                source_format=path.suffix.lower().lstrip("."),
                version=version,
                tree=indexing_result.tree_root,
                sections=sections,
                entity_map=indexing_result.entity_map,
            )

    async def reindex_document(self, file_path: str | Path, doc_id: str) -> DocumentIndex:
        path = Path(file_path)
        with self.logger.span("reindex_document", doc_id=doc_id, path=str(path)):
            if not self.store.has_document(doc_id):
                return await self.index_document(file_path, doc_id=doc_id)

            metadata = self.store.document_metadata(doc_id)
            old_sections = self.store.sections_for_doc(doc_id)
            new_sections = self.ingestion.parse(path, doc_id)
            diff = compute_section_diff(old_sections, new_sections)

            has_changes = bool(diff.changed_section_ids or diff.added_section_ids or diff.removed_section_ids)
            if not has_changes:
                version, tree = self.store.latest_tree(doc_id)
                self.logger.info("reindex_noop", doc_id=doc_id, version=version)
                return self._build_document_index(
                    doc_id=doc_id,
                    title=str(metadata["title"]),
                    source_format=str(metadata["source_format"]),
                    version=version,
                    tree=tree,
                    sections=new_sections,
                    entity_map={},
                )

            old_version, old_tree = self.store.latest_tree(doc_id)
            changed_nodes = map_changed_pages_to_nodes(old_tree, diff.changed_pages)
            change_ratio = len(diff.changed_pages) / max(1, len(new_sections))

            if change_ratio <= 0.5:
                rebuilt_tree = partial_rebuild_tree(
                    tree_builder=self.coordinator._tree_builder,
                    existing_root=old_tree,
                    all_sections=new_sections,
                    changed_pages=diff.changed_pages,
                    doc_id=doc_id,
                    title=str(metadata["title"]),
                )
                entity_pre, xrefs = self.entity_extractor.pre_extract(new_sections)
                entity_map, cross_refs = self.entity_extractor.finalize(
                    doc_id,
                    new_sections,
                    rebuilt_tree,
                    entity_pre,
                    xrefs,
                )
                graph_draft = self.graph_builder.prebuild(rebuilt_tree)
                graph_nodes, graph_edges = self.graph_builder.finalize(graph_draft, entity_map, cross_refs)
                self.store.remove_graph_edges_for_nodes(doc_id, changed_nodes)
                indexing_result = IndexingResult(
                    tree_root=rebuilt_tree,
                    entity_map=entity_map,
                    graph_nodes=graph_nodes,
                    graph_edges=graph_edges,
                )
                mode = "partial"
            else:
                indexing_result = await self.coordinator.build(doc_id, str(metadata["title"]), new_sections)
                mode = "full"

            indexing_result = self._augment_cross_doc_edges(source_doc_id=doc_id, result=indexing_result)

            version = self._save_index_with_retry(
                doc_id=doc_id,
                title=str(metadata["title"]),
                source_format=str(metadata["source_format"]),
                sections=new_sections,
                indexing_result=indexing_result,
                event_type="partial_reindex" if mode == "partial" else "full_reindex",
                event_details={
                    "mode": mode,
                    "old_version": old_version,
                    "changed_pages": sorted(diff.changed_pages),
                    "changed_nodes": sorted(changed_nodes),
                    "changed_sections": sorted(diff.changed_section_ids),
                    "added_sections": sorted(diff.added_section_ids),
                    "removed_sections": sorted(diff.removed_section_ids),
                },
            )

            self.logger.info(
                "reindex_completed",
                doc_id=doc_id,
                old_version=old_version,
                new_version=version,
                mode=mode,
                changed_pages=len(diff.changed_pages),
                changed_nodes=len(changed_nodes),
            )

            return self._build_document_index(
                doc_id=doc_id,
                title=str(metadata["title"]),
                source_format=str(metadata["source_format"]),
                version=version,
                tree=indexing_result.tree_root,
                sections=new_sections,
                entity_map=indexing_result.entity_map,
            )

    def query_document(self, doc_id: str, query: str, session_id: str | None = None) -> QueryResult:
        version = self.store.current_version(doc_id)
        idempotency_key = self._query_idempotency_key(doc_id, query, session_id, version=version)
        cached = self._query_idempotency_cache.get(idempotency_key)
        if cached is not None:
            self.logger.info("query_idempotent_hit", doc_id=doc_id, session_id=session_id)
            return cached.model_copy(deep=True)

        with self.logger.span("query_document", doc_id=doc_id, session_id=session_id):
            turns = self.store.recent_session_turns(session_id, limit=6) if session_id else []
            analysis = self.analyzer.analyze(query, turns)
            route = self.router.route(analysis)
            start = time.perf_counter()

            result = self._execute_query_route(route, doc_id, query, analysis)
            result.latency_ms = int((time.perf_counter() - start) * 1000)
            self._persist_query_result(result, session_id=session_id)
            self._run_adaptive_updates({source.doc_id for source in result.source_nodes} | {doc_id})
            self._query_idempotency_cache[idempotency_key] = result.model_copy(deep=True)
            return result

    def health(self) -> dict[str, Any]:
        status = self.health_checker.check(self.store)
        return {
            "ok": status.ok,
            "db_connected": status.db_connected,
            "docs_indexed": status.docs_indexed,
        }

    def _query_single_doc(self, doc_id: str, query: str, analysis: QueryAnalysis) -> QueryResult:
        version, tree_root = self.store.latest_tree(doc_id)
        cache_key = self._cache_key(doc_id, analysis.query_cluster_id, analysis.rewritten_query)
        cached = self.store.get_cache(cache_key, doc_id, version)

        entity_seed_nodes: set[str] = set()
        for entity in analysis.extracted_entities:
            entity_seed_nodes.update(self.store.entity_node_refs(entity, doc_id=doc_id))

        run = self.search.run(
            tree_root=tree_root,
            analysis=analysis,
            cached=cached["candidates"] if cached is not None else None,
            entity_seed_nodes=entity_seed_nodes,
        )
        if not run.cache_hit:
            self.store.put_cache(
                cache_key,
                doc_id,
                version,
                {"candidates": [candidate.model_dump() for candidate in run.candidates]},
            )

        chunks = self.retriever.retrieve(
            doc_id=doc_id,
            tree_root=tree_root,
            candidates=run.candidates,
            store=self.store,
        )
        answer, confidence, source_nodes = self.answer_generator.generate(analysis.rewritten_query, chunks)

        return QueryResult(
            query_id=str(uuid.uuid4()),
            original_query=query,
            rewritten_query=analysis.rewritten_query,
            intent_type="single_doc",
            traversal_path=run.traversal_path,
            source_nodes=source_nodes,
            answer=answer,
            confidence=confidence,
            latency_ms=0,
            tier1_calls=run.tier1_calls,
            tier2_calls=run.tier2_calls,
            cache_hit=run.cache_hit,
        )

    def _query_multi_doc(self, seed_doc_id: str, query: str, analysis: QueryAnalysis) -> QueryResult:
        all_docs = self.store.all_doc_ids()
        if seed_doc_id not in all_docs:
            all_docs.insert(0, seed_doc_id)
        candidates = [seed_doc_id, *[doc for doc in all_docs if doc != seed_doc_id]]
        selected_docs = candidates[:3]

        doc_roots = [
            (doc_id, self.store.latest_tree(doc_id)[1])
            for doc_id in selected_docs
            if self.store.has_document(doc_id)
        ]
        if not doc_roots:
            return self._query_single_doc(seed_doc_id, query, analysis)
        super_tree = build_virtual_super_tree(doc_roots)
        super_run = self.search.run(
            tree_root=super_tree,
            analysis=analysis,
            cached=None,
            entity_seed_nodes=set(),
        )

        doc_scores: dict[str, float] = {doc_id: 0.0 for doc_id in selected_docs}
        for candidate in super_run.candidates:
            for doc_id in selected_docs:
                if doc_id in candidate.node_id:
                    doc_scores[doc_id] = max(doc_scores[doc_id], candidate.confidence)

        ranked_docs = [
            doc
            for doc, _score in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
            if doc in selected_docs
        ]
        ranked_docs = ranked_docs[:2] if ranked_docs else selected_docs[:2]

        merged_sources = []
        merged_answer_lines = [f"Query: {analysis.rewritten_query}", "", "Cross-document summary:"]
        total_conf = 0.0
        tier1_calls = 0
        tier2_calls = 0

        for doc_id in ranked_docs:
            sub_result = self._query_single_doc(doc_id, query, analysis)
            tier1_calls += sub_result.tier1_calls
            tier2_calls += sub_result.tier2_calls
            total_conf += sub_result.confidence
            merged_sources.extend(sub_result.source_nodes)
            last_line = sub_result.answer.splitlines()[-1] if sub_result.answer else ""
            merged_answer_lines.append(f"- [{doc_id}] {last_line}")

        confidence = total_conf / max(1, len(ranked_docs))
        return QueryResult(
            query_id=str(uuid.uuid4()),
            original_query=query,
            rewritten_query=analysis.rewritten_query,
            intent_type="multi_doc",
            traversal_path=super_run.traversal_path,
            source_nodes=merged_sources[:6],
            answer="\n".join(merged_answer_lines),
            confidence=confidence,
            latency_ms=0,
            tier1_calls=tier1_calls,
            tier2_calls=tier2_calls,
            cache_hit=False,
        )

    def _query_graph(self, doc_id: str, query: str, analysis: QueryAnalysis) -> QueryResult:
        seed_refs: list[tuple[str, str]] = []
        for entity in analysis.extracted_entities:
            seed_refs.extend(self.store.entity_refs(entity))

        if not seed_refs:
            # No graph seeds, fallback to single doc path.
            return self._query_single_doc(doc_id, query, analysis)

        seed_keys = [f"{ref_doc}:{ref_node}" for ref_doc, ref_node in seed_refs]
        edge_types = self._edge_types_for_query(query)
        walked = self.graph_walker.walk(
            seed_node_keys=seed_keys,
            max_hops=3,
            edge_types=edge_types,
            bfs_fn=self.store.graph_bfs,
        )

        chunks = []
        traversal_path: list[str] = []
        for node_key in walked.visited_node_keys:
            if ":" not in node_key:
                continue
            node_doc, node_id = node_key.split(":", 1)
            if not self.store.has_document(node_doc):
                continue
            _version, tree = self.store.latest_tree(node_doc)
            node = find_node(tree, node_id)
            if node is None:
                continue
            traversal_path.append(node.node_id)
            candidates = [SearchCandidate(node_id=node.node_id, confidence=0.75)]
            chunks.extend(
                self.retriever.retrieve(
                    doc_id=node_doc,
                    tree_root=tree,
                    candidates=candidates,
                    store=self.store,
                )
            )

        answer, confidence, source_nodes = self.answer_generator.generate(analysis.rewritten_query, chunks)
        return QueryResult(
            query_id=str(uuid.uuid4()),
            original_query=query,
            rewritten_query=analysis.rewritten_query,
            intent_type="graph",
            traversal_path=traversal_path,
            source_nodes=source_nodes,
            answer=answer,
            confidence=confidence,
            latency_ms=0,
            tier1_calls=0,
            tier2_calls=0,
            cache_hit=False,
        )

    @with_retry(
        retries=2,
        delay_seconds=0.03,
        retry_exceptions=(RuntimeError, TimeoutError, OSError),
    )
    def _execute_query_route(
        self,
        route: str,
        doc_id: str,
        query: str,
        analysis: QueryAnalysis,
    ) -> QueryResult:
        if route == "multi_doc":
            return self._query_multi_doc(doc_id, query, analysis)
        if route == "graph":
            try:
                return self._query_graph(doc_id, query, analysis)
            except Exception as exc:
                self.logger.info("graph_fallback", doc_id=doc_id, reason=str(exc))
                return self._query_single_doc(doc_id, query, analysis)
        return self._query_single_doc(doc_id, query, analysis)

    def _persist_query_result(self, result: QueryResult, *, session_id: str | None) -> None:
        self.store.save_query_result(result, session_id=session_id)

        by_doc_visited: dict[str, list[str]] = {}
        by_doc_used: dict[str, list[str]] = {}
        for source in result.source_nodes:
            by_doc_used.setdefault(source.doc_id, []).append(source.node_id)
        for source in result.source_nodes:
            by_doc_visited.setdefault(source.doc_id, []).extend(result.traversal_path)

        if not by_doc_used:
            # fallback to first traversal doc unknown
            by_doc_used = {"unknown": []}
            by_doc_visited = {"unknown": result.traversal_path}

        for doc_key in by_doc_visited:
            if doc_key == "unknown":
                continue
            self.store.log_feedback_event(
                query_id=result.query_id,
                doc_id=doc_key,
                visited_nodes=by_doc_visited.get(doc_key, []),
                used_nodes=by_doc_used.get(doc_key, []),
                outcome_quality=None,
            )

        if session_id:
            self.store.add_session_turn(
                session_id=session_id,
                original_query=result.original_query,
                rewritten_query=result.rewritten_query,
                answer=result.answer,
            )

    def _run_adaptive_updates(self, doc_ids: set[str]) -> None:
        for doc_id in sorted(doc_ids):
            if not self.store.has_document(doc_id):
                continue
            feedback = [
                event
                for event in self.store.recent_feedback_events(limit=200)
                if event.get("doc_id") == doc_id
            ]
            if len(feedback) < 5:
                continue
            if len(feedback) % 5 != 0:
                continue
            _version, tree = self.store.latest_tree(doc_id)
            query_results = self.store.recent_query_results(limit=200)
            updated_tree, misleading = self.adaptive_updater.update_tree(
                doc_id=doc_id,
                tree_root=tree,
                query_results=query_results,
                feedback_events=feedback,
                store=self.store,
            )
            self.store.update_tree_for_document(doc_id, updated_tree, reason="adaptive_update")
            self.logger.info(
                "adaptive_update",
                doc_id=doc_id,
                misleading_nodes=len(misleading),
                feedback_events=len(feedback),
            )

    def _build_document_index(
        self,
        *,
        doc_id: str,
        title: str,
        source_format: str,
        version: int,
        tree: TreeNode,
        sections: list[Section],
        entity_map: dict[str, list[EntityRef]],
    ) -> DocumentIndex:
        total_nodes, max_depth, avg_depth = compute_tree_stats(tree)
        stats = DocumentStats(
            total_pages=max((section.page_number for section in sections), default=1),
            total_nodes=total_nodes,
            max_depth=max_depth,
            avg_depth=avg_depth,
        )
        return DocumentIndex(
            doc_id=doc_id,
            title=title,
            description=f"Index for {title}",
            source_format=source_format,
            version=version,
            tree=tree,
            entity_map=entity_map,
            stats=stats,
        )

    def _augment_cross_doc_edges(self, *, source_doc_id: str, result: IndexingResult) -> IndexingResult:
        other_docs = [doc for doc in self.store.all_doc_ids() if doc != source_doc_id]
        if not other_docs:
            return result

        source_nodes = [
            (source_doc_id, node.node_id, f"{node.title} {node.summary}")
            for node in walk_nodes(result.tree_root)
        ]
        other_nodes: list[tuple[str, str, str]] = []
        for doc_id in other_docs:
            if not self.store.has_document(doc_id):
                continue
            _v, tree = self.store.latest_tree(doc_id)
            other_nodes.extend(
                (doc_id, node.node_id, f"{node.title} {node.summary}")
                for node in walk_nodes(tree)
            )

        cross_edges = self.graph_builder.cross_doc_similarity_edges(
            source_doc_id=source_doc_id,
            source_nodes=source_nodes,
            other_docs_nodes=other_nodes,
        )
        if not cross_edges:
            return result

        merged_edges = self.graph_builder._dedupe_edges([*result.graph_edges, *cross_edges])
        result.graph_edges = merged_edges
        return result

    @with_retry(retries=3, delay_seconds=0.05)
    def _save_index_with_retry(
        self,
        *,
        doc_id: str,
        title: str,
        source_format: str,
        sections: list[Section],
        indexing_result: IndexingResult,
        event_type: str,
        event_details: dict[str, Any],
    ) -> int:
        return self.store.save_index(
            doc_id=doc_id,
            title=title,
            source_format=source_format,
            sections=sections,
            indexing_result=indexing_result,
            event_type=event_type,
            event_details=event_details,
        )

    @staticmethod
    def _edge_types_for_query(query: str) -> set[str]:
        q = query.lower()
        if "cite" in q or "reference" in q:
            return {"REFERENCES", "CROSS_DOC_REFERENCES"}
        if "compare" in q or "similar" in q:
            return {"CROSS_DOC_SIMILAR", "ELABORATES", "REFERENCES"}
        return {
            "ELABORATES",
            "REFERENCES",
            "TEMPORALLY_FOLLOWS",
            "CONTRADICTS",
            "CROSS_DOC_SIMILAR",
        }

    @staticmethod
    def _doc_id_from_path(path: Path) -> str:
        base = f"{path.stem}:{path.stat().st_size}:{path.resolve()}"
        digest = hashlib.sha1(base.encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"doc_{digest[:12]}"

    @staticmethod
    def _cache_key(doc_id: str, cluster_id: str, rewritten_query: str) -> str:
        payload = f"{doc_id}:{cluster_id}:{rewritten_query.lower().strip()}"
        return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()

    @staticmethod
    def _idempotency_key(action: str, doc_id: str, path: Path) -> str:
        payload = f"{action}:{doc_id}:{path.resolve()}:{path.stat().st_size}"
        return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()

    @staticmethod
    def _query_idempotency_key(
        doc_id: str,
        query: str,
        session_id: str | None,
        *,
        version: int,
    ) -> str:
        payload = f"{doc_id}:{session_id or '-'}:{query.strip().lower()}:{version}"
        return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


def build_engine(db_path: str | Path) -> QuerdexEngine:
    from querdex.llm import build_llm_client

    store = SQLiteStore(db_path)
    llm = build_llm_client()
    return QuerdexEngine(store, llm_client=llm)


def index_document(db_path: str | Path, file_path: str | Path, doc_id: str | None = None) -> DocumentIndex:
    engine = build_engine(db_path)
    try:
        return asyncio.run(engine.index_document(file_path, doc_id))
    finally:
        engine.store.close()
