from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class GraphWalkResult:
    visited_node_keys: list[str]
    visited_node_ids: list[str]


GraphBFSFn = Callable[..., list[str]]


class GraphWalker:
    """Graph traversal for relational queries (HI-078)."""

    def walk(
        self,
        *,
        seed_node_keys: list[str],
        max_hops: int,
        edge_types: set[str] | None,
        bfs_fn: GraphBFSFn,
    ) -> GraphWalkResult:
        visited_keys = bfs_fn(seed_nodes=seed_node_keys, max_hops=max_hops, edge_types=edge_types)
        visited_ids: list[str] = []
        for key in visited_keys:
            # format: doc_id:node_id
            if ":" in key:
                visited_ids.append(key.split(":", 1)[1])
            else:
                visited_ids.append(key)
        return GraphWalkResult(visited_node_keys=visited_keys, visited_node_ids=visited_ids)
