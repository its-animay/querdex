from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    source_format: str = Field(min_length=1)
    raw_bytes: bytes | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TreeNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    depth: int = Field(ge=0)
    affinity_scores: dict[str, float] = Field(default_factory=dict)
    visit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    utility_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=utc_now)
    children: list[TreeNode] = Field(default_factory=list)

    @field_validator("end_page")
    @classmethod
    def _validate_page_range(cls, value: int, info: Any) -> int:
        start_page = info.data.get("start_page")
        if start_page is not None and value < start_page:
            msg = "end_page must be greater than or equal to start_page"
            raise ValueError(msg)
        return value


class DocumentStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_pages: int = Field(ge=1)
    total_nodes: int = Field(ge=1)
    max_depth: int = Field(ge=0)
    avg_depth: float = Field(ge=0.0)


class EntityRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)


class DocumentIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_format: str = Field(min_length=1)
    version: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    tree: TreeNode
    entity_map: dict[str, list[EntityRef]] = Field(default_factory=dict)
    stats: DocumentStats


class SourceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    pages: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


IntentType = Literal["single_doc", "multi_doc", "graph"]


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    original_query: str = Field(min_length=1)
    rewritten_query: str = Field(min_length=1)
    intent_type: IntentType
    traversal_path: list[str] = Field(default_factory=list)
    source_nodes: list[SourceNode] = Field(default_factory=list)
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: int = Field(ge=0)
    tier1_calls: int = Field(ge=0)
    tier2_calls: int = Field(ge=0)
    cache_hit: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class QueryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_type: IntentType
    extracted_entities: list[str] = Field(default_factory=list)
    rewritten_query: str = Field(min_length=1)
    query_cluster_id: str = Field(min_length=1)


class SearchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class IndexingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tree_root: TreeNode
    entity_map: dict[str, list[EntityRef]]
    graph_nodes: list[tuple[str, str]]
    graph_edges: list[tuple[str, str, str, float]]
