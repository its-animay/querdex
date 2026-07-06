from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from querdex.schemas.models import utc_now

AlignmentStatus = Literal["exact", "fuzzy", "unaligned"]


class ExampleExtraction(BaseModel):
    """One expected extraction inside a few-shot example."""

    model_config = ConfigDict(extra="forbid")

    extraction_class: str = Field(min_length=1)
    extraction_text: str = Field(min_length=1)
    attributes: dict[str, str] = Field(default_factory=dict)


class ExtractionExample(BaseModel):
    """A few-shot example: sample text plus the extractions it should yield."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    extractions: list[ExampleExtraction] = Field(default_factory=list)


class ExtractionTask(BaseModel):
    """Schema-by-example task definition.

    The task is described in natural language and shaped by examples:
    the extraction classes and attribute keys used in ``examples`` define
    the output schema, without any hand-written parsing rules.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    examples: list[ExtractionExample] = Field(default_factory=list)


class Extraction(BaseModel):
    """A single extracted fact, grounded to its source span when possible.

    ``char_start``/``char_end`` are offsets into the *section content*
    identified by ``section_id``. ``alignment`` records how the span was
    located: ``exact`` (verbatim match), ``fuzzy`` (whitespace/similarity
    match), or ``unaligned`` (the model produced text that could not be
    found in the source - treat these as unverified).
    """

    model_config = ConfigDict(extra="forbid")

    extraction_class: str = Field(min_length=1)
    extraction_text: str = Field(min_length=1)
    attributes: dict[str, str] = Field(default_factory=dict)
    section_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    alignment: AlignmentStatus = "unaligned"
    pass_index: int = Field(default=0, ge=0)


class ExtractionStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_count: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    passes: int = Field(ge=1)
    exact_count: int = Field(default=0, ge=0)
    fuzzy_count: int = Field(default=0, ge=0)
    unaligned_count: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class ExtractionRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    task: ExtractionTask
    extractions: list[Extraction] = Field(default_factory=list)
    stats: ExtractionStats
    created_at: datetime = Field(default_factory=utc_now)
