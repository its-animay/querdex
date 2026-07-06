from __future__ import annotations

import json
from pathlib import Path

from querdex.extraction import (
    ExampleExtraction,
    ExtractionExample,
    ExtractionTask,
    StructuredExtractor,
    align,
    chunk_sections,
    render_extraction_html,
)
from querdex.llm.fake_client import FakeLLMClient
from querdex.schemas import Section
from querdex.storage import SQLiteStore


def _section(section_id: str, content: str, page: int = 1) -> Section:
    return Section(
        section_id=section_id,
        doc_id="doc_x",
        content=content,
        page_number=page,
        source_format="txt",
    )


def _task() -> ExtractionTask:
    return ExtractionTask(
        description="Extract revenue figures and executive names.",
        examples=[
            ExtractionExample(
                text="Alice Chen reported revenue of $5M in Q1.",
                extractions=[
                    ExampleExtraction(
                        extraction_class="metric",
                        extraction_text="revenue of $5M",
                        attributes={"period": "Q1"},
                    ),
                    ExampleExtraction(extraction_class="person", extraction_text="Alice Chen"),
                ],
            )
        ],
    )


# ---------------------------------------------------------------- chunker


def test_chunker_packs_sections_and_maps_offsets() -> None:
    sections = [_section("sec_0001", "First section."), _section("sec_0002", "Second section.", page=2)]
    chunks = chunk_sections(sections, max_chars=200)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "First section." in chunk.text
    assert "Second section." in chunk.text
    assert len(chunk.pieces) == 2

    second = chunk.pieces[1]
    assert second.section_id == "sec_0002"
    assert second.page_number == 2
    assert chunk.text[second.chunk_offset : second.chunk_offset + second.length] == "Second section."
    assert second.section_offset == 0


def test_chunker_splits_oversized_section_at_sentence_boundary() -> None:
    content = "One sentence here. " * 40  # ~760 chars
    chunks = chunk_sections([_section("sec_0001", content)], max_chars=300)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= 300
        for piece in chunk.pieces:
            original = content[piece.section_offset : piece.section_offset + piece.length]
            assert chunk.text[piece.chunk_offset : piece.chunk_offset + piece.length] == original


# ---------------------------------------------------------------- aligner


def test_align_exact_and_case_insensitive() -> None:
    assert align("revenue grew", "Total revenue grew 8%") == (6, 18, "exact")
    assert align("REVENUE GREW", "Total revenue grew 8%") == (6, 18, "exact")


def test_align_whitespace_normalized_is_fuzzy() -> None:
    result = align("revenue  grew", "Total revenue\n grew 8%")
    assert result is not None
    start, end, status = result
    assert status == "fuzzy"
    assert "revenue" in "Total revenue\n grew 8%"[start:end]


def test_align_rejects_unrelated_text() -> None:
    assert align("quantum entanglement", "Total revenue grew 8% in Q4") is None


# ---------------------------------------------------------------- extractor


def test_extractor_grounds_llm_extractions_to_sections() -> None:
    content = "In Q4 the company reported revenue of $12M. Bob Smith presented the results."
    fake = FakeLLMClient(
        default=json.dumps(
            {
                "extractions": [
                    {
                        "extraction_class": "metric",
                        "extraction_text": "revenue of $12M",
                        "attributes": {"period": "Q4"},
                    },
                    {"extraction_class": "person", "extraction_text": "Bob Smith"},
                ]
            }
        )
    )
    extractor = StructuredExtractor(llm_client=fake)
    run = extractor.extract(doc_id="doc_x", sections=[_section("sec_0001", content)], task=_task())

    assert len(run.extractions) == 2
    metric = next(e for e in run.extractions if e.extraction_class == "metric")
    assert metric.alignment == "exact"
    assert metric.section_id == "sec_0001"
    assert metric.page_number == 1
    assert metric.char_start is not None and metric.char_end is not None
    assert content[metric.char_start : metric.char_end] == "revenue of $12M"
    assert metric.attributes == {"period": "Q4"}
    assert run.stats.llm_calls == 1
    assert run.stats.exact_count == 2

    # Few-shot examples must be present in the prompt.
    assert "Alice Chen" in fake.calls[0]["user"]


def test_extractor_flags_hallucinated_text_as_unaligned() -> None:
    fake = FakeLLMClient(
        default=json.dumps(
            {"extractions": [{"extraction_class": "metric", "extraction_text": "profit of $99B"}]}
        )
    )
    extractor = StructuredExtractor(llm_client=fake)
    run = extractor.extract(doc_id="doc_x", sections=[_section("sec_0001", "Revenue was flat.")], task=_task())

    assert len(run.extractions) == 1
    assert run.extractions[0].alignment == "unaligned"
    assert run.extractions[0].section_id is None
    assert run.stats.unaligned_count == 1


def test_extractor_dedupes_across_passes() -> None:
    content = "Revenue of $12M was reported."
    fake = FakeLLMClient(
        default=json.dumps(
            {"extractions": [{"extraction_class": "metric", "extraction_text": "Revenue of $12M"}]}
        )
    )
    extractor = StructuredExtractor(llm_client=fake)
    run = extractor.extract(doc_id="doc_x", sections=[_section("sec_0001", content)], task=_task(), passes=2)

    assert len(run.extractions) == 1
    assert run.stats.llm_calls == 2
    assert run.stats.passes == 2


def test_extractor_survives_malformed_llm_output() -> None:
    fake = FakeLLMClient(default="Sorry, I cannot help with that.")
    extractor = StructuredExtractor(llm_client=fake)
    run = extractor.extract(doc_id="doc_x", sections=[_section("sec_0001", "Some content.")], task=_task())

    assert run.extractions == []
    assert run.stats.chunk_count == 1


def test_extractor_fallback_without_llm_matches_example_texts() -> None:
    content = "Quarterly update: revenue of $5M repeated, then revenue of $5M again."
    extractor = StructuredExtractor(llm_client=None)
    run = extractor.extract(doc_id="doc_x", sections=[_section("sec_0001", content)], task=_task())

    metrics = [e for e in run.extractions if e.extraction_class == "metric"]
    assert len(metrics) == 2
    assert all(e.alignment == "exact" for e in metrics)
    assert run.stats.llm_calls == 0


# ---------------------------------------------------------------- visualization


def test_render_html_highlights_and_escapes() -> None:
    content = "Revenue of $12M & <growth> was reported."
    fake = FakeLLMClient(
        default=json.dumps(
            {"extractions": [{"extraction_class": "metric", "extraction_text": "Revenue of $12M"}]}
        )
    )
    extractor = StructuredExtractor(llm_client=fake)
    sections = [_section("sec_0001", content)]
    run = extractor.extract(doc_id="doc_x", sections=sections, task=_task())

    html_out = render_extraction_html(run, sections)
    assert "<mark" in html_out
    assert "Revenue of $12M" in html_out
    assert "&lt;growth&gt;" in html_out  # source text is escaped
    assert "qx-legend" in html_out


def test_render_html_lists_ungrounded_extractions() -> None:
    fake = FakeLLMClient(
        default=json.dumps(
            {"extractions": [{"extraction_class": "metric", "extraction_text": "made-up numbers"}]}
        )
    )
    extractor = StructuredExtractor(llm_client=fake)
    sections = [_section("sec_0001", "Nothing to see here.")]
    run = extractor.extract(doc_id="doc_x", sections=sections, task=_task())

    html_out = render_extraction_html(run, sections)
    assert "Ungrounded extractions" in html_out
    assert "made-up numbers" in html_out


# ---------------------------------------------------------------- storage


def test_store_extraction_run_roundtrip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    try:
        extractor = StructuredExtractor(llm_client=None)
        run = extractor.extract(
            doc_id="doc_x",
            sections=[_section("sec_0001", "revenue of $5M closed the quarter.")],
            task=_task(),
        )
        store.save_extraction_run(run)

        loaded = store.get_extraction_run(run.run_id)
        assert loaded.doc_id == "doc_x"
        assert loaded.extractions == run.extractions

        runs = store.extraction_runs_for_doc("doc_x")
        assert [r.run_id for r in runs] == [run.run_id]
    finally:
        store.close()
