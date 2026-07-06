from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import networkx as nx  # type: ignore[import-untyped]


class NetworkXGraphStore:
    """Development graph adapter with JSON (node-link) persistence.

    JSON is used instead of pickle so that a shared or tampered graph file
    cannot execute code on load. Unreadable or legacy pickle files are
    ignored; the graph is derived data and is rebuilt on the next index pass.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = self._load()

    def _load(self) -> nx.DiGraph:
        if not self.path.exists():
            return nx.DiGraph()
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            loaded = nx.node_link_graph(data, directed=True, edges="edges")
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError, TypeError):
            return nx.DiGraph()
        if not isinstance(loaded, nx.DiGraph):
            return nx.DiGraph()
        return loaded

    def save(self) -> None:
        data = nx.node_link_data(self.graph, edges="edges")
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def upsert_document_graph(
        self,
        *,
        doc_id: str,
        nodes: list[tuple[str, str]],
        edges: list[tuple[str, str, str, float]],
    ) -> None:
        # Remove existing nodes for document first.
        for node in list(self.graph.nodes):
            node_doc = self.graph.nodes[node].get("doc_id")
            if node_doc == doc_id:
                self.graph.remove_node(node)

        for n_doc_id, node_id in nodes:
            key = f"{n_doc_id}:{node_id}"
            self.graph.add_node(key, doc_id=n_doc_id, node_id=node_id)

        for source, target, edge_type, weight in edges:
            source_key = f"{doc_id}:{source}" if ":" not in source else source
            target_key = f"{doc_id}:{target}" if ":" not in target else target
            self.graph.add_edge(source_key, target_key, edge_type=edge_type, weight=weight)

        self.save()

    def neighbors(
        self,
        node_key: str,
        *,
        edge_types: set[str] | None = None,
    ) -> list[str]:
        result: list[str] = []
        for _u, v, data in self.graph.out_edges(node_key, data=True):
            edge_type = str(data.get("edge_type", ""))
            if edge_types and edge_type not in edge_types:
                continue
            result.append(v)
        return result

    def bfs(
        self,
        *,
        seed_nodes: Iterable[str],
        max_hops: int = 3,
        edge_types: set[str] | None = None,
    ) -> list[str]:
        visited: set[str] = set()
        frontier = list(seed_nodes)
        ordered: list[str] = []

        for _ in range(max_hops + 1):
            if not frontier:
                break
            next_frontier: list[str] = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                ordered.append(node)
                next_frontier.extend(self.neighbors(node, edge_types=edge_types))
            frontier = next_frontier
        return ordered
