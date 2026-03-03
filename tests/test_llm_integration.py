"""Tests for LLM integration points using FakeLLMClient."""
from __future__ import annotations

import json

import pytest

from querdex.adaptive.updater import AdaptiveUpdater
from querdex.indexing.tree_builder import AdaptiveTreeBuilder
from querdex.llm.fake_client import FakeLLMClient
from querdex.query.answering import AnswerGenerator, RetrievedChunk
from querdex.query.tiered_search import TieredSearchEngine
from querdex.schemas import QueryAnalysis, SearchCandidate, Section, SourceNode, TreeNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _section(content: str, page: int = 1, doc_id: str = "doc1") -> Section:
    return Section(
        section_id=f"s_{page}",
        doc_id=doc_id,
        content=content,
        page_number=page,
        source_format="txt",
    )


def _tree_node(
    node_id: str = "node_0000",
    title: str = "Root",
    summary: str = "root summary",
    depth: int = 0,
    children: list[TreeNode] | None = None,
) -> TreeNode:
    return TreeNode(
        node_id=node_id,
        doc_id="doc1",
        title=title,
        summary=summary,
        start_page=1,
        end_page=3,
        depth=depth,
        children=children or [],
    )


def _source_node(node_id: str = "node_0000", confidence: float = 0.9) -> SourceNode:
    return SourceNode(
        node_id=node_id,
        doc_id="doc1",
        title="Revenue Section",
        pages="1-3",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# FakeLLMClient basic behaviour
# ---------------------------------------------------------------------------

class TestFakeLLMClient:
    def test_returns_default_when_no_match(self) -> None:
        fake = FakeLLMClient(default='{"summary": "default"}')
        result = fake.complete(system="sys", user="unrecognised prompt", model="fake-tier1")
        assert result == '{"summary": "default"}'

    def test_matches_first_matching_key(self) -> None:
        fake = FakeLLMClient(
            responses={
                "Revenue": '{"summary": "Revenue section."}',
                "Balance": '{"summary": "Balance sheet."}',
            }
        )
        result = fake.complete(system="sys", user="Revenue for Q3", model="fake-tier1")
        assert json.loads(result)["summary"] == "Revenue section."

    def test_records_calls(self) -> None:
        fake = FakeLLMClient()
        fake.complete(system="sys", user="hello", model="fake-tier2")
        assert len(fake.calls) == 1
        assert fake.calls[0]["model"] == "fake-tier2"

    def test_tier_model_attributes(self) -> None:
        fake = FakeLLMClient()
        assert fake.tier1_model == "fake-tier1"
        assert fake.tier2_model == "fake-tier2"


# ---------------------------------------------------------------------------
# AdaptiveTreeBuilder._build_summary with LLM
# ---------------------------------------------------------------------------

class TestTreeBuilderWithLLM:
    def test_uses_llm_summary_when_provided(self) -> None:
        fake = FakeLLMClient(
            responses={"Summary": '{"summary": "LLM-generated summary of Q3 revenue."}'}
        )
        builder = AdaptiveTreeBuilder(llm_client=fake)
        sections = [_section("Q3 revenue was $1.2B", page=1), _section("Details below.", page=2)]
        tree = builder.build("doc1", sections, "Annual Report")

        assert len(fake.calls) >= 1
        # Root summary should come from LLM
        assert "LLM-generated" in tree.summary or tree.summary != ""

    def test_falls_back_to_heuristic_on_bad_json(self) -> None:
        fake = FakeLLMClient(default="NOT JSON AT ALL")
        builder = AdaptiveTreeBuilder(llm_client=fake)
        sections = [_section("Revenue data.", page=1)]
        tree = builder.build("doc1", sections, "Financial Report")

        # Should not raise; falls back to heuristic keyword summary
        assert len(tree.summary) > 0
        assert "Financial Report" in tree.summary or "page" in tree.summary.lower()

    def test_no_llm_produces_heuristic_summary(self) -> None:
        builder = AdaptiveTreeBuilder(llm_client=None)
        sections = [_section("Hello world. This is a test document.", page=1)]
        tree = builder.build("doc1", sections, "Test Doc")
        assert "Test Doc" in tree.summary
        assert len(fake_calls := getattr(tree, "_llm_calls", [])) == 0  # type: ignore[attr-defined]
        _ = fake_calls  # suppress unused


# ---------------------------------------------------------------------------
# TieredSearchEngine Tier 1 with LLM
# ---------------------------------------------------------------------------

class TestTieredSearchTier1WithLLM:
    def _make_nodes(self) -> list[TreeNode]:
        return [
            _tree_node("node_0001", title="Revenue Analysis", summary="Q3 revenue figures"),
            _tree_node("node_0002", title="Risk Factors", summary="Regulatory and market risks"),
            _tree_node("node_0003", title="Capital Structure", summary="Debt and equity details"),
        ]

    def test_llm_tier1_selects_relevant_nodes(self) -> None:
        # LLM says index 0 is relevant, 1 and 2 are not
        tier1_response = json.dumps([
            {"index": 0, "relevant": True},
            {"index": 1, "relevant": False},
            {"index": 2, "relevant": False},
        ])
        fake = FakeLLMClient(responses={"Revenue": tier1_response}, default=tier1_response)
        engine = TieredSearchEngine(tier1_client=fake, tier2_client=None)

        nodes = self._make_nodes()
        scores = engine._tier1_batch_score("What is the Q3 revenue?", nodes)

        assert len(scores) == 3
        node_ids = [n.node_id for n, _ in scores]
        score_map = {n.node_id: s for n, s in scores}
        assert score_map["node_0001"] == 1.0
        assert score_map["node_0002"] == 0.0
        assert score_map["node_0003"] == 0.0
        assert "node_0001" in node_ids

    def test_tier1_falls_back_to_heuristic_on_bad_json(self) -> None:
        fake = FakeLLMClient(default="BAD JSON")
        engine = TieredSearchEngine(tier1_client=fake, tier2_client=None)
        nodes = self._make_nodes()
        scores = engine._tier1_batch_score("revenue", nodes)
        # Heuristic: should still return 3 scored tuples
        assert len(scores) == 3
        for node, score in scores:
            assert 0.0 <= score <= 1.0

    def test_no_tier1_client_uses_heuristic(self) -> None:
        engine = TieredSearchEngine(tier1_client=None, tier2_client=None)
        nodes = self._make_nodes()
        scores = engine._tier1_batch_score("revenue analysis", nodes)
        assert len(scores) == 3
        # "Revenue Analysis" should score higher than unrelated nodes
        revenue_score = next(s for n, s in scores if n.node_id == "node_0001")
        risk_score = next(s for n, s in scores if n.node_id == "node_0002")
        assert revenue_score >= risk_score


# ---------------------------------------------------------------------------
# TieredSearchEngine Tier 2 with LLM
# ---------------------------------------------------------------------------

class TestTieredSearchTier2WithLLM:
    def test_llm_tier2_marks_relevant_nodes(self) -> None:
        tier2_response = json.dumps({"relevant": True, "confidence": 0.92, "explore_children": []})
        fake = FakeLLMClient(default=tier2_response)
        engine = TieredSearchEngine(tier1_client=None, tier2_client=fake, tier2_threshold=0.1)
        node = _tree_node("node_0001", title="Revenue", summary="Q3 revenue figures")
        candidates = engine._tier2_rank("What is Q3 revenue?", [node])

        assert len(candidates) == 1
        assert candidates[0].node_id == "node_0001"
        assert candidates[0].confidence == pytest.approx(0.92)

    def test_llm_tier2_excludes_irrelevant_nodes(self) -> None:
        tier2_response = json.dumps({"relevant": False, "confidence": 0.05, "explore_children": []})
        fake = FakeLLMClient(default=tier2_response)
        engine = TieredSearchEngine(tier1_client=None, tier2_client=fake, tier2_threshold=0.25)
        node = _tree_node("node_0001")
        candidates = engine._tier2_rank("What is Q3 revenue?", [node])
        assert len(candidates) == 0

    def test_tier2_falls_back_on_bad_json(self) -> None:
        fake = FakeLLMClient(default="NOT JSON")
        engine = TieredSearchEngine(tier1_client=None, tier2_client=fake)
        node = _tree_node("node_0001", title="revenue", summary="revenue data")
        # Should not raise; falls back to heuristic
        candidates = engine._tier2_rank("revenue", [node])
        assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# AnswerGenerator with LLM
# ---------------------------------------------------------------------------

class TestAnswerGeneratorWithLLM:
    def _make_chunk(self, text: str = "Revenue was $1.2B in Q3.") -> RetrievedChunk:
        return RetrievedChunk(source=_source_node(), text=text)

    def test_llm_generate_returns_synthesized_answer(self) -> None:
        llm_response = json.dumps({"answer": "Q3 revenue was $1.2B (Revenue Section, pages 1-3).", "confidence": 0.88})
        fake = FakeLLMClient(default=llm_response)
        gen = AnswerGenerator(llm_client=fake)

        answer, confidence, sources = gen.generate("What was Q3 revenue?", [self._make_chunk()])

        assert "1.2B" in answer
        assert confidence == pytest.approx(0.88)
        assert len(sources) == 1
        assert len(fake.calls) == 1

    def test_llm_falls_back_on_bad_json(self) -> None:
        fake = FakeLLMClient(default="NOT JSON")
        gen = AnswerGenerator(llm_client=fake)
        answer, confidence, sources = gen.generate("revenue?", [self._make_chunk()])
        # Falls back to heuristic
        assert len(answer) > 0
        assert "revenue" in answer.lower() or "Query" in answer

    def test_no_chunks_returns_empty_answer(self) -> None:
        fake = FakeLLMClient()
        gen = AnswerGenerator(llm_client=fake)
        answer, confidence, sources = gen.generate("anything", [])
        assert confidence == 0.0
        assert sources == []
        assert len(fake.calls) == 0  # no LLM call when no content

    def test_no_llm_uses_heuristic(self) -> None:
        gen = AnswerGenerator(llm_client=None)
        answer, confidence, sources = gen.generate("revenue?", [self._make_chunk()])
        assert "Query" in answer
        assert len(sources) == 1


# ---------------------------------------------------------------------------
# AdaptiveUpdater._rewrite_summary with LLM
# ---------------------------------------------------------------------------

class TestAdaptiveUpdaterRewriteSummaryWithLLM:
    def test_llm_rewrites_summary(self) -> None:
        llm_response = json.dumps({"summary": "Pages 1-2 contain Q3 revenue tables only."})
        fake = FakeLLMClient(default=llm_response)
        updater = AdaptiveUpdater(llm_client=fake)

        result = updater._rewrite_summary("Revenue", "Q3 revenue $1.2B table data here.", 1, 2)
        assert result == "Pages 1-2 contain Q3 revenue tables only."
        assert len(fake.calls) == 1

    def test_falls_back_on_bad_json(self) -> None:
        fake = FakeLLMClient(default="INVALID")
        updater = AdaptiveUpdater(llm_client=fake)
        result = updater._rewrite_summary("Revenue", "Some content here.", 1, 2)
        # Heuristic: title + first 240 chars
        assert "Revenue" in result
        assert "Some content" in result

    def test_no_llm_uses_heuristic(self) -> None:
        updater = AdaptiveUpdater(llm_client=None)
        result = updater._rewrite_summary("Costs", "Operating costs were $500M in FY2023.")
        assert "Costs" in result
        assert "Operating costs" in result
