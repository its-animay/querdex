from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from querdex.schemas import QueryResult, Section, TreeNode


def test_section_schema_validates() -> None:
    section = Section(
        section_id="sec_0001",
        doc_id="doc_1",
        content="Hello world",
        page_number=1,
        source_format="markdown",
        metadata={"heading": "Intro"},
    )
    assert section.section_id == "sec_0001"


def test_section_schema_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        Section(
            section_id="sec_0001",
            doc_id="doc_1",
            content="",
            page_number=1,
            source_format="markdown",
            metadata={},
        )


def test_tree_node_page_range_validation() -> None:
    with pytest.raises(ValidationError):
        TreeNode(
            node_id="node_1",
            doc_id="doc_1",
            title="Bad",
            summary="Bad",
            start_page=4,
            end_page=2,
            depth=0,
            last_updated=datetime.now(UTC),
            children=[],
        )


def test_query_result_schema() -> None:
    result = QueryResult(
        query_id="q1",
        original_query="hello",
        rewritten_query="hello",
        intent_type="single_doc",
        answer="ok",
        confidence=0.5,
        latency_ms=10,
        tier1_calls=1,
        tier2_calls=1,
    )
    assert result.intent_type == "single_doc"
