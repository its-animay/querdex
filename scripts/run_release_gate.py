from __future__ import annotations

import argparse
import asyncio
import json
import sys

from hyperindex.evaluation import EvaluationHarness, KPIDashboardBuilder, RetrievalMetricsHarness
from hyperindex.services import build_engine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HyperIndex evaluation and release gate checks.")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--doc-path", required=True, help="Path to document to index for the benchmark run")
    parser.add_argument("--doc-id", required=True, help="Document id used for benchmark queries")
    parser.add_argument("--fixture", default="tests/fixtures/eval/baseline_cases.json", help="Evaluation fixture JSON")
    parser.add_argument("--out", default="index_store/kpi_dashboard.json", help="Output KPI dashboard JSON")

    parser.add_argument("--min-intent-accuracy", type=float, default=0.0)
    parser.add_argument("--min-citation-coverage", type=float, default=0.0)
    parser.add_argument("--max-p95-latency-ms", type=float, default=2000.0)
    parser.add_argument("--max-avg-cost-usd", type=float, default=0.01)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    engine = build_engine(args.db)
    harness = EvaluationHarness()
    metrics = RetrievalMetricsHarness()
    dashboard_builder = KPIDashboardBuilder()

    try:
        asyncio.run(engine.index_document(args.doc_path, doc_id=args.doc_id))
        cases = harness.load_cases(args.fixture)
        results = [engine.query_document(args.doc_id, case.query, session_id="release_gate") for case in cases]

        eval_summary = harness.score_results(cases, results)
        retrieval_summary = metrics.summarize(results)
        dashboard = dashboard_builder.build(
            eval_summary=eval_summary,
            retrieval_summary=retrieval_summary,
        )
        dashboard_builder.write_json(args.out, dashboard)
        print(json.dumps(dashboard, indent=2))

        failed = []
        if eval_summary.intent_accuracy < args.min_intent_accuracy:
            failed.append("intent_accuracy")
        if eval_summary.citation_coverage < args.min_citation_coverage:
            failed.append("citation_coverage")
        if retrieval_summary.p95_latency_ms > args.max_p95_latency_ms:
            failed.append("p95_latency_ms")
        if retrieval_summary.avg_cost_usd > args.max_avg_cost_usd:
            failed.append("avg_cost_usd")

        if failed:
            print(f"Release gate failed on: {', '.join(failed)}", file=sys.stderr)
            return 1
        return 0
    finally:
        engine.store.close()


if __name__ == "__main__":
    raise SystemExit(main())
