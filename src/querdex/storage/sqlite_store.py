from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from querdex.indexing import EntityMapUpdater
from querdex.schemas import EntityRef, IndexingResult, QueryResult, Section, TreeNode
from querdex.storage.graph_store import NetworkXGraphStore


class SQLiteStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._entity_map_updater = EntityMapUpdater()
        self._graph_store = NetworkXGraphStore(Path(self.db_path).with_suffix(".graph.pkl"))
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS documents (
              doc_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              source_format TEXT NOT NULL,
              version INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tree_versions (
              doc_id TEXT NOT NULL,
              version INTEGER NOT NULL,
              tree_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY (doc_id, version)
            );

            CREATE TABLE IF NOT EXISTS sections (
              doc_id TEXT NOT NULL,
              section_id TEXT NOT NULL,
              page_number INTEGER NOT NULL,
              content TEXT NOT NULL,
              source_format TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              raw_bytes BLOB,
              PRIMARY KEY (doc_id, section_id)
            );

            CREATE TABLE IF NOT EXISTS entity_map (
              entity TEXT NOT NULL,
              doc_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              version INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_entity_map_entity ON entity_map(entity);
            CREATE INDEX IF NOT EXISTS idx_entity_map_doc_id ON entity_map(doc_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_map_unique ON entity_map(entity, doc_id, node_id);

            CREATE TABLE IF NOT EXISTS graph_nodes (
              doc_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              version INTEGER NOT NULL,
              PRIMARY KEY (doc_id, node_id, version)
            );

            CREATE TABLE IF NOT EXISTS graph_edges (
              doc_id TEXT NOT NULL,
              source_node_id TEXT NOT NULL,
              target_node_id TEXT NOT NULL,
              edge_type TEXT NOT NULL,
              weight REAL NOT NULL,
              version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS version_history (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              doc_id TEXT NOT NULL,
              version INTEGER NOT NULL,
              event_type TEXT NOT NULL,
              details_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS query_results (
              query_id TEXT PRIMARY KEY,
              session_id TEXT,
              result_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              query_id TEXT NOT NULL,
              doc_id TEXT NOT NULL,
              visited_nodes_json TEXT NOT NULL,
              used_nodes_json TEXT NOT NULL,
              outcome_quality REAL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS node_metrics (
              doc_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              visit_count INTEGER NOT NULL DEFAULT 0,
              use_count INTEGER NOT NULL DEFAULT 0,
              last_updated TEXT NOT NULL,
              PRIMARY KEY (doc_id, node_id)
            );

            CREATE TABLE IF NOT EXISTS affinity_scores (
              doc_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              cluster_id TEXT NOT NULL,
              score REAL NOT NULL,
              last_updated TEXT NOT NULL,
              PRIMARY KEY (doc_id, node_id, cluster_id)
            );

            CREATE TABLE IF NOT EXISTS summary_regen_queue (
              queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
              doc_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              prompt TEXT NOT NULL,
              status TEXT NOT NULL,
              new_summary TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT NOT NULL,
              turn_id INTEGER NOT NULL,
              original_query TEXT NOT NULL,
              rewritten_query TEXT NOT NULL,
              answer TEXT NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY (session_id, turn_id)
            );

            CREATE TABLE IF NOT EXISTS query_cache (
              cache_key TEXT PRIMARY KEY,
              doc_id TEXT NOT NULL,
              doc_version INTEGER NOT NULL,
              value_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(tz=UTC).isoformat()

    def current_version(self, doc_id: str) -> int:
        row = self._conn.execute(
            "SELECT version FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return int(row["version"]) if row else 0

    def has_document(self, doc_id: str) -> bool:
        return self.current_version(doc_id) > 0

    def latest_tree(self, doc_id: str) -> tuple[int, TreeNode]:
        row = self._conn.execute(
            """
            SELECT version, tree_json
            FROM tree_versions
            WHERE doc_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (doc_id,),
        ).fetchone()
        if row is None:
            msg = f"No indexed tree found for doc_id={doc_id}"
            raise KeyError(msg)
        return int(row["version"]), TreeNode.model_validate_json(str(row["tree_json"]))

    def save_index(
        self,
        *,
        doc_id: str,
        title: str,
        source_format: str,
        sections: list[Section],
        indexing_result: IndexingResult,
        event_type: str = "index_build",
        event_details: dict[str, Any] | None = None,
    ) -> int:
        now = self._now()
        next_version = self.current_version(doc_id) + 1

        self._conn.execute(
            """
            INSERT INTO documents(doc_id, title, source_format, version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                title=excluded.title,
                source_format=excluded.source_format,
                version=excluded.version,
                updated_at=excluded.updated_at
            """,
            (doc_id, title, source_format, next_version, now, now),
        )

        self._conn.execute(
            "INSERT INTO tree_versions(doc_id, version, tree_json, created_at) VALUES (?, ?, ?, ?)",
            (
                doc_id,
                next_version,
                indexing_result.tree_root.model_dump_json(),
                now,
            ),
        )

        # HI-051b: section content store.
        self._conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
        for section in sections:
            self._conn.execute(
                """
                INSERT INTO sections(
                    doc_id, section_id, page_number, content, source_format, metadata_json, raw_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    section.doc_id,
                    section.section_id,
                    section.page_number,
                    section.content,
                    section.source_format,
                    json.dumps(section.metadata),
                    section.raw_bytes,
                ),
            )

        self._persist_entity_map_incremental(doc_id=doc_id, version=next_version, incoming=indexing_result.entity_map)

        self._conn.execute("DELETE FROM graph_nodes WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM graph_edges WHERE doc_id = ?", (doc_id,))
        for n_doc_id, node_id in indexing_result.graph_nodes:
            self._conn.execute(
                "INSERT INTO graph_nodes(doc_id, node_id, version) VALUES (?, ?, ?)",
                (n_doc_id, node_id, next_version),
            )
        for source, target, edge_type, weight in indexing_result.graph_edges:
            self._conn.execute(
                """
                INSERT INTO graph_edges(doc_id, source_node_id, target_node_id, edge_type, weight, version)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, source, target, edge_type, weight, next_version),
            )
        self._graph_store.upsert_document_graph(
            doc_id=doc_id,
            nodes=indexing_result.graph_nodes,
            edges=indexing_result.graph_edges,
        )

        self._conn.execute(
            """
            INSERT INTO version_history(doc_id, version, event_type, details_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                next_version,
                event_type,
                json.dumps(
                    event_details
                    or {"section_count": len(sections), "entity_count": len(indexing_result.entity_map)}
                ),
                now,
            ),
        )

        self.invalidate_cache_for_document(doc_id)
        self._conn.commit()
        return next_version

    def _persist_entity_map_incremental(
        self,
        *,
        doc_id: str,
        version: int,
        incoming: dict[str, list[EntityRef]],
    ) -> None:
        # For re-index of same doc we replace old doc rows; for new docs this appends globally.
        self._conn.execute("DELETE FROM entity_map WHERE doc_id = ?", (doc_id,))
        rows = self._entity_map_updater.to_rows(incoming, version=version)
        self._conn.executemany(
            "INSERT OR IGNORE INTO entity_map(entity, doc_id, node_id, version) VALUES (?, ?, ?, ?)",
            rows,
        )

    def fetch_sections_by_page_range(self, doc_id: str, start_page: int, end_page: int) -> list[Section]:
        rows = self._conn.execute(
            """
            SELECT section_id, doc_id, page_number, content, source_format, metadata_json, raw_bytes
            FROM sections
            WHERE doc_id = ? AND page_number BETWEEN ? AND ?
            ORDER BY page_number ASC
            """,
            (doc_id, start_page, end_page),
        ).fetchall()
        return [
            Section(
                section_id=str(row["section_id"]),
                doc_id=str(row["doc_id"]),
                page_number=int(row["page_number"]),
                content=str(row["content"]),
                source_format=str(row["source_format"]),
                metadata=json.loads(str(row["metadata_json"])),
                raw_bytes=row["raw_bytes"],
            )
            for row in rows
        ]

    def entity_node_refs(self, entity: str, doc_id: str | None = None) -> list[str]:
        if doc_id is None:
            rows = self._conn.execute(
                "SELECT DISTINCT node_id FROM entity_map WHERE entity = ?",
                (entity,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT node_id FROM entity_map WHERE entity = ? AND doc_id = ?",
                (entity, doc_id),
            ).fetchall()
        return [str(row["node_id"]) for row in rows]

    def entity_refs(self, entity: str) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT DISTINCT doc_id, node_id FROM entity_map WHERE entity = ?",
            (entity,),
        ).fetchall()
        return [(str(row["doc_id"]), str(row["node_id"])) for row in rows]

    def all_doc_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT doc_id FROM documents ORDER BY doc_id ASC").fetchall()
        return [str(row["doc_id"]) for row in rows]

    def document_metadata(self, doc_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT doc_id, title, source_format, version, created_at, updated_at FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            msg = f"Document not found: {doc_id}"
            raise KeyError(msg)
        return {
            "doc_id": str(row["doc_id"]),
            "title": str(row["title"]),
            "source_format": str(row["source_format"]),
            "version": int(row["version"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def sections_for_doc(self, doc_id: str) -> list[Section]:
        rows = self._conn.execute(
            """
            SELECT section_id, doc_id, page_number, content, source_format, metadata_json, raw_bytes
            FROM sections
            WHERE doc_id = ?
            ORDER BY page_number ASC, section_id ASC
            """,
            (doc_id,),
        ).fetchall()
        return [
            Section(
                section_id=str(row["section_id"]),
                doc_id=str(row["doc_id"]),
                page_number=int(row["page_number"]),
                content=str(row["content"]),
                source_format=str(row["source_format"]),
                metadata=json.loads(str(row["metadata_json"])),
                raw_bytes=row["raw_bytes"],
            )
            for row in rows
        ]

    def save_query_result(self, query_result: QueryResult, session_id: str | None = None) -> None:
        # HI-092b: persistent query results for adaptive loop inputs.
        self._conn.execute(
            """
            INSERT OR REPLACE INTO query_results(query_id, session_id, result_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                query_result.query_id,
                session_id,
                query_result.model_dump_json(),
                self._now(),
            ),
        )
        self._conn.commit()

    def recent_query_results(self, limit: int = 100) -> list[QueryResult]:
        rows = self._conn.execute(
            "SELECT result_json FROM query_results ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [QueryResult.model_validate_json(str(row["result_json"])) for row in rows]

    def update_tree_for_document(self, doc_id: str, tree: TreeNode, *, reason: str = "tree_update") -> int:
        now = self._now()
        next_version = self.current_version(doc_id) + 1
        self._conn.execute(
            "UPDATE documents SET version = ?, updated_at = ? WHERE doc_id = ?",
            (next_version, now, doc_id),
        )
        self._conn.execute(
            "INSERT INTO tree_versions(doc_id, version, tree_json, created_at) VALUES (?, ?, ?, ?)",
            (doc_id, next_version, tree.model_dump_json(), now),
        )
        self._conn.execute(
            """
            INSERT INTO version_history(doc_id, version, event_type, details_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, next_version, reason, json.dumps({"reason": reason}), now),
        )
        self.invalidate_cache_for_document(doc_id)
        self._conn.commit()
        return next_version

    def log_feedback_event(
        self,
        *,
        query_id: str,
        doc_id: str,
        visited_nodes: list[str],
        used_nodes: list[str],
        outcome_quality: float | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO feedback_events(
                query_id, doc_id, visited_nodes_json, used_nodes_json, outcome_quality, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                doc_id,
                json.dumps(visited_nodes),
                json.dumps(used_nodes),
                outcome_quality,
                self._now(),
            ),
        )
        self._conn.commit()

    def recent_feedback_events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT query_id, doc_id, visited_nodes_json, used_nodes_json, outcome_quality, created_at
            FROM feedback_events
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "query_id": str(row["query_id"]),
                "doc_id": str(row["doc_id"]),
                "visited_nodes": cast(list[str], json.loads(str(row["visited_nodes_json"]))),
                "used_nodes": cast(list[str], json.loads(str(row["used_nodes_json"]))),
                "outcome_quality": None if row["outcome_quality"] is None else float(row["outcome_quality"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def upsert_node_metric(self, *, doc_id: str, node_id: str, visited: int, used: int) -> None:
        now = self._now()
        self._conn.execute(
            """
            INSERT INTO node_metrics(doc_id, node_id, visit_count, use_count, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, node_id) DO UPDATE SET
                visit_count = node_metrics.visit_count + excluded.visit_count,
                use_count = node_metrics.use_count + excluded.use_count,
                last_updated = excluded.last_updated
            """,
            (doc_id, node_id, visited, used, now),
        )
        self._conn.commit()

    def node_metrics_for_doc(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT doc_id, node_id, visit_count, use_count, last_updated
            FROM node_metrics
            WHERE doc_id = ?
            ORDER BY node_id ASC
            """,
            (doc_id,),
        ).fetchall()
        return [
            {
                "doc_id": str(row["doc_id"]),
                "node_id": str(row["node_id"]),
                "visit_count": int(row["visit_count"]),
                "use_count": int(row["use_count"]),
                "last_updated": str(row["last_updated"]),
            }
            for row in rows
        ]

    def upsert_affinity_score(self, *, doc_id: str, node_id: str, cluster_id: str, score: float) -> None:
        now = self._now()
        self._conn.execute(
            """
            INSERT INTO affinity_scores(doc_id, node_id, cluster_id, score, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, node_id, cluster_id) DO UPDATE SET
                score = excluded.score,
                last_updated = excluded.last_updated
            """,
            (doc_id, node_id, cluster_id, score, now),
        )
        self._conn.commit()

    def affinity_scores_for_doc(self, doc_id: str) -> dict[str, dict[str, float]]:
        rows = self._conn.execute(
            """
            SELECT node_id, cluster_id, score
            FROM affinity_scores
            WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchall()
        result: dict[str, dict[str, float]] = {}
        for row in rows:
            node_id = str(row["node_id"])
            cluster = str(row["cluster_id"])
            score = float(row["score"])
            result.setdefault(node_id, {})[cluster] = score
        return result

    def enqueue_summary_regen(self, *, doc_id: str, node_id: str, prompt: str) -> None:
        now = self._now()
        self._conn.execute(
            """
            INSERT INTO summary_regen_queue(doc_id, node_id, prompt, status, created_at, updated_at)
            VALUES (?, ?, ?, 'queued', ?, ?)
            """,
            (doc_id, node_id, prompt, now, now),
        )
        self._conn.commit()

    def pending_summary_regen(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT queue_id, doc_id, node_id, prompt, status, new_summary
            FROM summary_regen_queue
            WHERE status = 'queued'
            ORDER BY queue_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "queue_id": int(row["queue_id"]),
                "doc_id": str(row["doc_id"]),
                "node_id": str(row["node_id"]),
                "prompt": str(row["prompt"]),
                "status": str(row["status"]),
                "new_summary": None if row["new_summary"] is None else str(row["new_summary"]),
            }
            for row in rows
        ]

    def mark_summary_regen_done(self, *, queue_id: int, new_summary: str) -> None:
        self._conn.execute(
            """
            UPDATE summary_regen_queue
            SET status = 'done', new_summary = ?, updated_at = ?
            WHERE queue_id = ?
            """,
            (new_summary, self._now(), queue_id),
        )
        self._conn.commit()

    def graph_neighbors(self, node_key: str, *, edge_types: set[str] | None = None) -> list[str]:
        return self._graph_store.neighbors(node_key, edge_types=edge_types)

    def graph_bfs(
        self,
        *,
        seed_nodes: list[str],
        max_hops: int = 3,
        edge_types: set[str] | None = None,
    ) -> list[str]:
        return self._graph_store.bfs(seed_nodes=seed_nodes, max_hops=max_hops, edge_types=edge_types)

    def remove_graph_edges_for_nodes(self, doc_id: str, node_ids: set[str]) -> None:
        if not node_ids:
            return
        placeholders = ",".join("?" for _ in node_ids)
        params = [doc_id, *node_ids, *node_ids]
        self._conn.execute(
            f"""
            DELETE FROM graph_edges
            WHERE doc_id = ? AND (source_node_id IN ({placeholders}) OR target_node_id IN ({placeholders}))
            """,
            params,
        )
        self._conn.commit()

    def add_session_turn(
        self,
        session_id: str,
        original_query: str,
        rewritten_query: str,
        answer: str,
    ) -> None:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(turn_id), 0) + 1 AS next_turn FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        turn_id = int(row["next_turn"]) if row is not None else 1
        self._conn.execute(
            """
            INSERT INTO sessions(session_id, turn_id, original_query, rewritten_query, answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, turn_id, original_query, rewritten_query, answer, self._now()),
        )
        self._conn.commit()

    def recent_session_turns(self, session_id: str, limit: int = 6) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT turn_id, original_query, rewritten_query, answer, created_at
            FROM sessions
            WHERE session_id = ?
            ORDER BY turn_id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [
            {
                "turn_id": int(row["turn_id"]),
                "original_query": str(row["original_query"]),
                "rewritten_query": str(row["rewritten_query"]),
                "answer": str(row["answer"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def put_cache(self, cache_key: str, doc_id: str, doc_version: int, value: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO query_cache(cache_key, doc_id, doc_version, value_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cache_key, doc_id, doc_version, json.dumps(value), self._now()),
        )
        self._conn.commit()

    def get_cache(self, cache_key: str, doc_id: str, doc_version: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT value_json
            FROM query_cache
            WHERE cache_key = ? AND doc_id = ? AND doc_version = ?
            """,
            (cache_key, doc_id, doc_version),
        ).fetchone()
        if row is None:
            return None
        parsed = json.loads(str(row["value_json"]))
        return cast(dict[str, Any], parsed)

    def invalidate_cache_for_document(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM query_cache WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def delete_document(self, doc_id: str) -> None:
        # HI-057b: cleanup across all stores.
        self._conn.execute("DELETE FROM tree_versions WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM entity_map WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM graph_nodes WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM graph_edges WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM query_cache WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._graph_store.upsert_document_graph(doc_id=doc_id, nodes=[], edges=[])
        self._conn.execute(
            """
            INSERT INTO version_history(doc_id, version, event_type, details_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, 0, "document_deleted", json.dumps({"doc_id": doc_id}), self._now()),
        )
        self._conn.commit()
